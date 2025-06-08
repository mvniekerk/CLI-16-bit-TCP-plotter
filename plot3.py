import socket
import struct
import threading
import time
import argparse
import logging
from collections import deque
import queue

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration (CRITICAL: Adjust these to match your bushtelegram_dsp setup) ---
SAMPLE_SIZE = 256  # Number of f32 samples to read at a time for FFT
SAMPLING_RATE = 936  # Hz - !!! IMPORTANT: SET THIS TO YOUR ACTUAL SAMPLING RATE !!!
# For a real input of SAMPLE_SIZE, np.fft.rfft gives SAMPLE_SIZE//2 + 1 complex values.
# This is the length of the FFT magnitude array BEFORE slicing.
FFT_RAW_OUTPUT_LEN = SAMPLE_SIZE // 2 + 1

SPLIT_AT_LEFT = 2  # Bins to discard from the start of FFT_RAW_OUTPUT_LEN
SPLIT_AT_RIGHT = 45 # Bins to discard from the end of FFT_RAW_OUTPUT_LEN

# Calculated length of the data that will actually be plotted for FFT
PLOT_DATA_LENGTH_FFT = FFT_RAW_OUTPUT_LEN - SPLIT_AT_LEFT - SPLIT_AT_RIGHT

# Footsteps graph parameters
FOOTSTEPS_FREQ_MIN = 8  # Hz
FOOTSTEPS_FREQ_MAX = 20 # Hz
FOOTSTEPS_TRUNCATE_VAL = 5.0
FOOTSTEPS_Y_MAX = 50.0
# --- End of Critical Configuration ---

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# Thread-safe queue to pass data (raw_samples, fft_data) from TCP thread to plotting
# raw_samples will be ignored by the plotting logic in this version, but kept for minimal changes to TCP thread
data_queue = queue.Queue(maxsize=5)  # Store a few recent datasets to avoid lag

class PlotStatistics:
    """Holds statistics for plotting, similar to Rust App state (primarily for FFT)."""
    def __init__(self):
        self.overall_max_value_fft = 0.0
        self.frame_averages_history_fft = deque(maxlen=20) # For moving average of FFT frame averages
        self.current_moving_avg_of_frame_avgs_fft = 0.0
        self.current_frame_max_value_fft = 0.0

stats = PlotStatistics()

# --- Pre-calculate Footsteps Bin Indices ---
# These are indices into the *original* (unsliced) FFT output
DELTA_F = SAMPLING_RATE  / (SAMPLE_SIZE)
FOOTSTEPS_BIN_START_ORIG = int(round(FOOTSTEPS_FREQ_MIN / DELTA_F))
FOOTSTEPS_BIN_END_ORIG = int(round(FOOTSTEPS_FREQ_MAX / DELTA_F))

logging.info(f"Sampling Rate: {SAMPLING_RATE} Hz, Sample Size: {SAMPLE_SIZE}")
logging.info(f"FFT Bin Resolution (Delta F): {DELTA_F:.2f} Hz")
logging.info(f"Footsteps original bin range: {FOOTSTEPS_BIN_START_ORIG} to {FOOTSTEPS_BIN_END_ORIG}")
# ---

def tcp_data_receiver_thread(server_host, server_port):
    """
    Connects to TCP server, reads SAMPLE_SIZE floats, performs FFT,
    slices FFT data, and puts (raw_samples, processed_fft_data) onto the data_queue.
    """
    while True: # Outer loop for attempting reconnections
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                logging.info(f"Attempting to connect to {server_host}:{server_port}...")
                sock.connect((server_host, server_port))
                logging.info(f"Connected to {server_host}:{server_port}")

                while True:
                    bytes_to_read = SAMPLE_SIZE * 4  # Each f32 is 4 bytes
                    raw_bytes = b''
                    while len(raw_bytes) < bytes_to_read:
                        chunk = sock.recv(bytes_to_read - len(raw_bytes))
                        if not chunk:
                            logging.error("Connection closed by server.")
                            raise ConnectionResetError("Server closed connection")
                        raw_bytes += chunk

                    try:
                        samples = np.array(struct.unpack(f'<{SAMPLE_SIZE}f', raw_bytes), dtype=np.float32)
                    except struct.error as e:
                        logging.error(f"Error unpacking data: {e}. Read {len(raw_bytes)} bytes, expected {bytes_to_read}.")
                        continue

                    fft_complex_result = np.fft.rfft(samples)
                    fft_magnitudes = np.abs(fft_complex_result)

                    if len(fft_magnitudes) != FFT_RAW_OUTPUT_LEN:
                        logging.warning(
                            f"FFT output length mismatch. Expected {FFT_RAW_OUTPUT_LEN}, "
                            f"got {len(fft_magnitudes)}. Check SAMPLE_SIZE."
                        )
                        current_fft_len = len(fft_magnitudes)
                    else:
                        current_fft_len = FFT_RAW_OUTPUT_LEN

                    if SPLIT_AT_LEFT + SPLIT_AT_RIGHT >= current_fft_len:
                        logging.error(
                            f"Slicing parameters ({SPLIT_AT_LEFT}, {SPLIT_AT_RIGHT}) "
                            f"are too large for FFT output length ({current_fft_len}). No data to plot for FFT."
                        )
                        processed_fft_data = np.array([])
                    else:
                        processed_fft_data = fft_magnitudes[SPLIT_AT_LEFT : current_fft_len - SPLIT_AT_RIGHT]

                    final_fft_data_for_plot = processed_fft_data.astype(np.float64)
                    data_to_plot = (samples.copy(), final_fft_data_for_plot) # raw samples are included but not used by plot
                    try:
                        data_queue.put(data_to_plot, block=False)
                    except queue.Full:
                        try:
                            data_queue.get_nowait()
                            data_queue.put_nowait(data_to_plot)
                        except queue.Empty: pass
                        except queue.Full: logging.warning("Data queue remained full after trying to make space.")

        except (ConnectionRefusedError, ConnectionResetError, socket.timeout, OSError) as e:
            logging.error(f"TCP connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Unexpected error in TCP receiver thread: {e}", exc_info=True)
            time.sleep(5)


# --- Matplotlib Plotting Setup ---
fig, (ax_fft, ax_footsteps) = plt.subplots(2, 1, figsize=(10, 8), sharex=False) # 2 rows, 1 column

# Plot for FFT Data (Top Plot)
line_fft_data, = ax_fft.plot([], [], lw=1.5, color='cyan', label='FFT Data')
line_moving_avg_fft, = ax_fft.plot([], [], lw=1, color='yellow', linestyle='--', label='Moving Avg (FFT Frame Avgs)')
stats_text_display = ax_fft.text(0.01, 0.01, "", fontsize=9, va='bottom', ha='left', transform=ax_fft.transAxes)

# Plot for Footsteps Data (Bottom Plot)
line_footsteps, = ax_footsteps.plot([], [], lw=1.5, color='magenta', label=f'Footsteps ({FOOTSTEPS_FREQ_MIN}-{FOOTSTEPS_FREQ_MAX}Hz)')
text_footsteps_value = ax_footsteps.text(0.5, -0.15, "", fontsize=10, ha='center', va='top', transform=ax_footsteps.transAxes)


actual_plotted_length_fft = PLOT_DATA_LENGTH_FFT

def init_animation():
    global actual_plotted_length_fft

    # Initialize FFT Plot
    ax_fft.set_xlim(0, actual_plotted_length_fft -1 if actual_plotted_length_fft > 0 else 1)
    ax_fft.set_ylim(0, 100)
    ax_fft.set_title("FFT Magnitude Spectrum")
    ax_fft.set_xlabel("Frequency Bin Index (Post-Slice)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.legend(loc='upper right')
    ax_fft.grid(True, linestyle=':', alpha=0.7)
    line_fft_data.set_data([], [])
    line_moving_avg_fft.set_data([], [])
    stats_text_display.set_text("")

    # Initialize Footsteps Plot
    ax_footsteps.set_ylim(0, FOOTSTEPS_Y_MAX)
    ax_footsteps.set_title(f"Footsteps ({FOOTSTEPS_FREQ_MIN}Hz - {FOOTSTEPS_FREQ_MAX}Hz, Truncated at {FOOTSTEPS_TRUNCATE_VAL})")
    ax_footsteps.set_xlabel("Relative Bin Index in Selected Range")
    ax_footsteps.set_ylabel("Truncated Magnitude")
    ax_footsteps.legend(loc='upper right')
    ax_footsteps.grid(True, linestyle=':', alpha=0.7)
    line_footsteps.set_data([], [])
    text_footsteps_value.set_text("")
    # X-lim for footsteps will be set dynamically in update_plot_frame

    fig.suptitle("Live FFT and Footsteps Data from TCP Stream", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect bottom for footsteps text

    return line_fft_data, line_moving_avg_fft, stats_text_display, line_footsteps, text_footsteps_value

def update_plot_frame(frame_number):
    global actual_plotted_length_fft
    try:
        _, current_fft_values = data_queue.get_nowait() # raw_samples are ignored
    except queue.Empty:
        return line_fft_data, line_moving_avg_fft, stats_text_display, line_footsteps, text_footsteps_value

    # --- Update FFT Plot ---
    if current_fft_values is None or len(current_fft_values) == 0:
        line_fft_data.set_data([], [])
        line_moving_avg_fft.set_data([], [])
        line_footsteps.set_data([], []) # Also clear footsteps if FFT is empty
        text_footsteps_value.set_text("Sum: N/A")
        # Don't update FFT stats if no FFT data
        return line_fft_data, line_moving_avg_fft, stats_text_display, line_footsteps, text_footsteps_value

    actual_plotted_length_fft = len(current_fft_values)
    x_fft_indices = np.arange(actual_plotted_length_fft)

    # Update FFT Statistics
    if actual_plotted_length_fft > 0:
        stats.current_frame_max_value_fft = np.max(current_fft_values)
        current_frame_average_fft = np.mean(current_fft_values)
    else:
        stats.current_frame_max_value_fft = 0.0
        current_frame_average_fft = 0.0

    if stats.current_frame_max_value_fft > stats.overall_max_value_fft:
        stats.overall_max_value_fft = stats.current_frame_max_value_fft
    stats.frame_averages_history_fft.append(current_frame_average_fft)
    if stats.frame_averages_history_fft:
        stats.current_moving_avg_of_frame_avgs_fft = np.mean(list(stats.frame_averages_history_fft))

    # Update FFT Plot Data
    line_fft_data.set_data(x_fft_indices, current_fft_values)
    if actual_plotted_length_fft > 0:
        line_moving_avg_fft.set_data(
            [0, actual_plotted_length_fft - 1],
            [stats.current_moving_avg_of_frame_avgs_fft, stats.current_moving_avg_of_frame_avgs_fft]
        )
    else:
        line_moving_avg_fft.set_data([],[])

    # Update FFT Plot Limits
    y_lower_fft = stats.current_moving_avg_of_frame_avgs_fft / 2.0
    y_upper_fft = stats.overall_max_value_fft
    if y_upper_fft <= y_lower_fft: y_upper_fft = y_lower_fft + 1.0
    y_range_fft = y_upper_fft - y_lower_fft
    y_padding_fft = y_range_fft * 0.1 if y_range_fft > 0 else 0.5
    ax_fft.set_xlim(0, actual_plotted_length_fft - 1 if actual_plotted_length_fft > 1 else 1)
    ax_fft.set_ylim(max(0, y_lower_fft - y_padding_fft), y_upper_fft + y_padding_fft)

    # Update Statistics Text (FFT)
    stats_str = (
        f"FFT Frame Max: {stats.current_frame_max_value_fft:.2f}\n"
        f"FFT Overall Max: {stats.overall_max_value_fft:.2f}\n"
        f"FFT Mov.Avg (Frame Avgs): {stats.current_moving_avg_of_frame_avgs_fft:.2f}"
    )
    stats_text_display.set_text(stats_str)

    # --- Update Footsteps Plot ---
    # Map original bin indices to the sliced `current_fft_values`
    # current_fft_values corresponds to original bins from SPLIT_AT_LEFT to FFT_RAW_OUTPUT_LEN - SPLIT_AT_RIGHT - 1

    start_idx_in_sliced = max(0, FOOTSTEPS_BIN_START_ORIG - SPLIT_AT_LEFT)
    # +1 because FOOTSTEPS_BIN_END_ORIG is inclusive, and slicing is exclusive for the end
    end_idx_in_sliced = FOOTSTEPS_BIN_END_ORIG - SPLIT_AT_LEFT + 1
    end_idx_in_sliced = min(len(current_fft_values), end_idx_in_sliced)


    if start_idx_in_sliced < end_idx_in_sliced and start_idx_in_sliced < len(current_fft_values):
        footsteps_bins_data = current_fft_values[start_idx_in_sliced:end_idx_in_sliced]
        truncated_footsteps_bins = np.minimum(footsteps_bins_data, FOOTSTEPS_TRUNCATE_VAL)
        footsteps_metric_sum = np.sum(truncated_footsteps_bins)

        x_footsteps_indices = np.arange(len(truncated_footsteps_bins))
        line_footsteps.set_data(x_footsteps_indices, truncated_footsteps_bins)
        ax_footsteps.set_xlim(0, len(x_footsteps_indices) - 1 if len(x_footsteps_indices) > 1 else 1)
        text_footsteps_value.set_text(f"Sum in Range: {footsteps_metric_sum:.2f}")
    else:
        # No valid bins for footsteps in the current FFT data (e.g., due to slicing or FFT length)
        line_footsteps.set_data([], [])
        ax_footsteps.set_xlim(0, 1) # Default xlim
        text_footsteps_value.set_text("Sum in Range: N/A (No bins)")
        logging.debug(f"No valid bins for footsteps. Start_idx_sliced: {start_idx_in_sliced}, End_idx_sliced: {end_idx_in_sliced}, len_fft: {len(current_fft_values)}")


    return line_fft_data, line_moving_avg_fft, stats_text_display, line_footsteps, text_footsteps_value

def main():
    parser = argparse.ArgumentParser(description="Connects to a TCP socket, processes, and plots FFT and Footsteps data.")
    parser.add_argument(
        "-s", "--server", default="192.168.88.250:1234", # Changed default to localhost for easier testing
        help="The server address and port (e.g., 192.168.88.250:1234)"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Verbosity level (e.g., -v for DEBUG, -vv for more)."
    )
    args = parser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)

    if PLOT_DATA_LENGTH_FFT <= 0:
        logging.error(
            f"Calculated PLOT_DATA_LENGTH_FFT is {PLOT_DATA_LENGTH_FFT}, which is not valid for FFT plot. "
            f"Check SAMPLE_SIZE ({SAMPLE_SIZE}), FFT_RAW_OUTPUT_LEN ({FFT_RAW_OUTPUT_LEN}), "
            f"SPLIT_AT_LEFT ({SPLIT_AT_LEFT}), and SPLIT_AT_RIGHT ({SPLIT_AT_RIGHT})."
        )
        return

    if FOOTSTEPS_BIN_START_ORIG > FOOTSTEPS_BIN_END_ORIG:
        logging.warning(f"Footsteps min frequency results in a higher bin index ({FOOTSTEPS_BIN_START_ORIG}) "
                        f"than max frequency ({FOOTSTEPS_BIN_END_ORIG}). Check SAMPLING_RATE and FREQ settings.")

    if FOOTSTEPS_BIN_END_ORIG >= FFT_RAW_OUTPUT_LEN :
        logging.warning(f"Footsteps max bin ({FOOTSTEPS_BIN_END_ORIG}) is out of raw FFT range ({FFT_RAW_OUTPUT_LEN-1}). "
                        f"May result in empty footsteps plot.")


    try:
        server_host, server_port_str = args.server.split(':')
        server_port = int(server_port_str)
    except ValueError:
        logging.error("Invalid server format. Use HOST:PORT (e.g., 192.168.88.250:1234)")
        return

    receiver = threading.Thread(
        target=tcp_data_receiver_thread,
        args=(server_host, server_port),
        daemon=True,
        name="TCPReceiverThread"
    )
    receiver.start()

    ani = animation.FuncAnimation(
        fig,
        update_plot_frame,
        init_func=init_animation,
        interval=100, #ms
        blit=True,
        cache_frame_data=False
    )

    plt.show()
    logging.info("Plot window closed. Application exiting.")

if __name__ == "__main__":
    main()