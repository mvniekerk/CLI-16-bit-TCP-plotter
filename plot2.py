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
# SAMPLE_SIZE = 256  # Number of f32 samples to read at a time for FFT
SAMPLE_SIZE = 128  # Number of f32 samples to read at a time for FFT
# For a real input of SAMPLE_SIZE, np.fft.rfft gives SAMPLE_SIZE//2 + 1 complex values.
# This is the length of the FFT magnitude array BEFORE slicing.
FFT_RAW_OUTPUT_LEN = SAMPLE_SIZE // 2 + 1

SPLIT_AT_LEFT = 2  # Bins to discard from the start of FFT_RAW_OUTPUT_LEN
SPLIT_AT_RIGHT = 50 # Bins to discard from the end of FFT_RAW_OUTPUT_LEN

# Calculated length of the data that will actually be plotted for FFT
PLOT_DATA_LENGTH_FFT = FFT_RAW_OUTPUT_LEN - SPLIT_AT_LEFT - SPLIT_AT_RIGHT
# --- End of Critical Configuration ---

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# Thread-safe queue to pass data (raw_samples, fft_data) from TCP thread to plotting
data_queue = queue.Queue(maxsize=5)  # Store a few recent datasets to avoid lag

class PlotStatistics:
    """Holds statistics for plotting, similar to Rust App state (primarily for FFT)."""
    def __init__(self):
        self.overall_max_value_fft = 0.0
        self.frame_averages_history_fft = deque(maxlen=20) # For moving average of FFT frame averages
        self.current_moving_avg_of_frame_avgs_fft = 0.0
        self.current_frame_max_value_fft = 0.0
        self.current_frame_sum = 0.0

stats = PlotStatistics()

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

                    # Unpack as little-endian floats
                    try:
                        # Keep raw samples as np.float32, as received
                        samples = np.array(struct.unpack(f'<{SAMPLE_SIZE}f', raw_bytes), dtype=np.float32)
                    except struct.error as e:
                        logging.error(f"Error unpacking data: {e}. Read {len(raw_bytes)} bytes, expected {bytes_to_read}.")
                        continue # Skip this corrupted/incomplete packet

                    # Apply FFT (real FFT, then magnitudes)
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

                    # Slice the FFT output
                    if SPLIT_AT_LEFT + SPLIT_AT_RIGHT >= current_fft_len:
                        logging.error(
                            f"Slicing parameters ({SPLIT_AT_LEFT}, {SPLIT_AT_RIGHT}) "
                            f"are too large for FFT output length ({current_fft_len}). No data to plot for FFT."
                        )
                        processed_fft_data = np.array([])
                    else:
                        processed_fft_data = fft_magnitudes[SPLIT_AT_LEFT : current_fft_len - SPLIT_AT_RIGHT]

                    # Convert FFT data to float64 for consistency if needed, raw samples remain f32
                    final_fft_data_for_plot = processed_fft_data.astype(np.float64)

                    # Put data onto the queue for the plotting function
                    # Store a tuple: (raw_samples_for_wave_plot, final_fft_data_for_fft_plot)
                    data_to_plot = (samples.copy(), final_fft_data_for_plot)
                    try:
                        data_queue.put(data_to_plot, block=False) # Non-blocking put
                    except queue.Full:
                        try:
                            data_queue.get_nowait() # Make space by removing oldest
                            data_queue.put_nowait(data_to_plot)
                        except queue.Empty:
                            pass
                        except queue.Full:
                            logging.warning("Data queue remained full after trying to make space.")

        except (ConnectionRefusedError, ConnectionResetError, socket.timeout, OSError) as e:
            logging.error(f"TCP connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Unexpected error in TCP receiver thread: {e}", exc_info=True)
            time.sleep(5)


# --- Matplotlib Plotting Setup ---
# Create a figure with two subplots, one for raw wave, one for FFT
fig, (ax_wave, ax_fft) = plt.subplots(2, 1, figsize=(10, 8)) # 2 rows, 1 column

# Plot for Raw Waveform
line_raw_wave, = ax_wave.plot([], [], lw=1.0, color='lime', label='Raw Waveform')

# Plot for FFT Data
line_fft_data, = ax_fft.plot([], [], lw=1.5, color='cyan', label='FFT Data')
line_moving_avg_fft, = ax_fft.plot([], [], lw=1, color='yellow', linestyle='--', label='Moving Avg (FFT Frame Avgs)')

# Text for displaying statistics on the plot (associated with FFT plot for now)
stats_text_display = ax_fft.text(0.01, 0.01, "", fontsize=9, va='bottom', ha='left', transform=ax_fft.transAxes)

# Store the actual length of data being plotted for FFT x-axis scaling
actual_plotted_length_fft = PLOT_DATA_LENGTH_FFT

def init_animation():
    """Initializes the plot elements for animation."""
    global actual_plotted_length_fft

    # Initialize FFT Plot
    ax_fft.set_xlim(0, actual_plotted_length_fft -1 if actual_plotted_length_fft > 0 else 1)
    ax_fft.set_ylim(0, 50) # Initial Y-limits, will be adjusted dynamically
    ax_fft.set_title("FFT Magnitude Spectrum")
    ax_fft.set_xlabel("Frequency Bin Index (Post-Slice)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.legend(loc='upper right')
    ax_fft.grid(True, linestyle=':', alpha=0.7)
    line_fft_data.set_data([], [])
    line_moving_avg_fft.set_data([], [])

    fig.suptitle("Live Waveform and FFT Data from TCP Stream", fontsize=14)
    # Adjust layout for suptitle, stats_text_display, and space between subplots
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    stats_text_display.set_text("")
    return line_raw_wave, line_fft_data, line_moving_avg_fft, stats_text_display

def update_plot_frame(frame_number):
    """Called by FuncAnimation to update the plot for each frame."""
    global actual_plotted_length_fft
    try:
        # Get the latest data from the queue: (raw_samples, fft_values)
        raw_samples, current_fft_values = data_queue.get_nowait()
    except queue.Empty:
        # No new data, just return the existing artists
        return line_raw_wave, line_fft_data, line_moving_avg_fft, stats_text_display

    # --- Update FFT Plot ---
    if current_fft_values is None or len(current_fft_values) == 0:
        # If FFT data is empty, clear its lines but keep raw wave if it exists
        line_fft_data.set_data([], [])
        line_moving_avg_fft.set_data([], [])
        # Don't update FFT stats if no FFT data
        return line_raw_wave, line_fft_data, line_moving_avg_fft, stats_text_display

    actual_plotted_length_fft = len(current_fft_values)
    x_fft_indices = np.arange(actual_plotted_length_fft)

    # --- Update FFT Statistics ---
    if actual_plotted_length_fft > 0:
        stats.current_frame_max_value_fft = np.max(current_fft_values)
        stats.current_frame_sum = np.sum(current_fft_values)
        current_frame_average_fft = np.mean(current_fft_values)
    else:
        stats.current_frame_max_value_fft = 0.0
        stats.current_frame_sum = 0.0
        current_frame_average_fft = 0.0

    if stats.current_frame_max_value_fft > stats.overall_max_value_fft:
        stats.overall_max_value_fft = stats.current_frame_max_value_fft

    stats.frame_averages_history_fft.append(current_frame_average_fft)
    if stats.frame_averages_history_fft:
        stats.current_moving_avg_of_frame_avgs_fft = np.mean(list(stats.frame_averages_history_fft))

    # --- Update FFT Plot Data ---
    line_fft_data.set_data(x_fft_indices, current_fft_values)

    if actual_plotted_length_fft > 0:
        line_moving_avg_fft.set_data(
            [0, actual_plotted_length_fft - 1],
            [stats.current_moving_avg_of_frame_avgs_fft, stats.current_moving_avg_of_frame_avgs_fft]
        )
    else:
        line_moving_avg_fft.set_data([],[])

    # --- Update FFT Plot Limits (Y-axis dynamically, X-axis if length changes) ---
    y_lower_fft = stats.current_moving_avg_of_frame_avgs_fft / 2.0
    y_upper_fft = stats.overall_max_value_fft

    if y_upper_fft <= y_lower_fft:
        y_upper_fft = y_lower_fft + 1.0
    y_range_fft = y_upper_fft - y_lower_fft
    y_padding_fft = y_range_fft * 0.1 if y_range_fft > 0 else 0.5

    ax_fft.set_xlim(0, actual_plotted_length_fft - 1 if actual_plotted_length_fft > 1 else 1)
    # ax_fft.set_ylim(max(0, y_lower_fft - y_padding_fft), y_upper_fft + y_padding_fft)
    ax_fft.set_ylim(0, 50)

    # --- Update Statistics Text (related to FFT) ---
    stats_str = (
        f"FFT Frame Max: {stats.current_frame_max_value_fft:.2f}\n"
        f"FFT Sum: {stats.current_frame_sum:.2f}\n"
        f"FFT Overall Max: {stats.overall_max_value_fft:.2f}\n"
        f"FFT Mov.Avg (Frame Avgs): {stats.current_moving_avg_of_frame_avgs_fft:.2f}"
    )
    stats_text_display.set_text(stats_str)

    return line_raw_wave, line_fft_data, line_moving_avg_fft, stats_text_display

def main():
    parser = argparse.ArgumentParser(description="Connects to a TCP socket, processes, and plots raw wave and FFT data.")
    parser.add_argument(
        "-s", "--server", default="192.168.88.250:1234",
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
        # We can still proceed if raw wave plot is desired, but FFT plot will be empty/problematic.
        # For now, let's return if FFT plot length is invalid, as it's a core part.
        return

    try:
        server_host, server_port_str = args.server.split(':')
        server_port = int(server_port_str)
    except ValueError:
        logging.error("Invalid server format. Use HOST:PORT (e.g., 127.0.0.1:1234)")
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
        interval=100,
        blit=True,
        cache_frame_data=False
    )

    plt.show()
    logging.info("Plot window closed. Application exiting.")

if __name__ == "__main__":
    main()