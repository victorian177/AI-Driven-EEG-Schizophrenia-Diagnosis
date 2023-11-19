import mne

for i in range(1, 6):
    # Specify the path to your EDF file
    edf_file_path = f"Sample EDF Files/Phase {i}.edf"

    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)

    # Plot the raw data
    fig = raw.plot(scalings="auto", title="EDF Data")

    # Save the plot to a file (e.g., PNG)
    fig.savefig(f"Output EDF Plots/edf_plot_{i}.png")
