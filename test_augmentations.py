import numpy as np
from scipy.io import wavfile
import librosa
import soundfile as sf


def add_white_noise(audio, noise_var_coeff):
    """Adds zero mean Gaussian noise to audio with specified variance."""
    coeff = noise_var_coeff * np.mean(np.abs(audio))
    noisy_audio = audio + coeff * np.random.randn(len(audio))
    return noisy_audio


def shift(audio, shift_sec, fs):
    """Shifts audio."""
    shift_count = int(shift_sec * fs)
    return np.roll(audio, shift_count)


def stretch(audio, rate=1):
    """Stretches audio with specified ratio."""
    input_length = 16000
    audio = audio.astype(float)  # Convert audio to floating-point
    audio2 = librosa.effects.time_stretch(audio, rate=rate)
    if len(audio2) > input_length:
        audio2 = audio2[:input_length]
    else:
        audio2 = np.pad(audio2, (0, max(0, input_length - len(audio2))), "constant")

    return audio2


def pitch_shift(audio, fs, shift_steps):
    """Shifts the pitch of the audio signal by the specified number of steps.

    Args:
        audio (ndarray): Input audio signal.
        fs (int): Sampling frequency of the audio.
        shift_steps (float): Number of steps to shift the pitch. Positive values increase the pitch, negative values decrease the pitch.

    Returns:
        ndarray: Audio signal with pitch shifted.
    """
    return librosa.effects.pitch_shift(audio.astype(float), sr=fs, n_steps=shift_steps)


def add_echo(audio, fs, delay, decay):
    """Adds echo effect to audio with specified delay and decay.

    Args:
        audio (ndarray): Input audio signal.
        fs (int): Sampling frequency of the audio.
        delay (float): Delay time in seconds.
        decay (float): Decay factor for the echo effect.

    Returns:
        ndarray: Audio signal with echo effect added.
    """
    # Calculate the number of samples that correspond to the delay time
    delay_samples = int(delay * fs)

    # Set the decay factor for the echo
    decay_factor = decay

    # Create a new array of zeros with the same shape as the audio signal
    echo_audio = np.zeros_like(audio)

    # Add a scaled, delayed version of the audio signal to the echo_audio array
    # This creates the echo effect
    echo_audio[delay_samples:] = audio[:-delay_samples] * decay_factor

    # Add the original audio signal to the echo signal and return the result
    # This combines the original signal with the echo
    return audio + echo_audio


def apply_reverberation(audio, fs, room_size, reverberation_time):
    """Applies reverberation effect to audio with specified room size and reverberation time.

    Args:
        audio (ndarray): Input audio signal.
        fs (int): Sampling frequency of the audio.
        room_size (float): Room size parameter for the reverberation effect.
        reverberation_time (float): Reverberation time parameter for the reverberation effect.

    Returns:
        ndarray: Audio signal with reverberation effect added.
    """
    # Calculate the number of samples that correspond to the reverberation time
    reverberation_samples = int(reverberation_time * fs)

    # Generate the impulse response for the reverberation effect
    impulse_response = np.random.randn(reverberation_samples)

    # Apply the impulse response to the audio signal using convolution
    reverberated_audio = np.convolve(audio, impulse_response, mode="same")

    # Scale the reverberated audio to avoid clipping
    max_value = np.max(np.abs(reverberated_audio))
    if max_value > 1.0:
        reverberated_audio /= max_value

    # Adjust the room size parameter to control the amount of reverberation
    reverberated_audio *= room_size

    return reverberated_audio


# Read the .wav audio file
audio_file_path = "speech_commands_v0.02/right/2a0b413e_nohash_0.wav"
file_name = audio_file_path.split("/")[-1]
fs, audio = wavfile.read(audio_file_path)


# Apply the echo effect
# Define the echo parameters
delay = 0.3  # delay in seconds
decay = 0.3  # decay factor

audio_with_echo = add_echo(audio, fs, delay, decay)

# Apply the pitch shift effect
shift_steps = float(2)  # Example value, adjust as needed
audio_with_pitch_shift = pitch_shift(audio, fs, shift_steps)

# Apply the reverberation effect
room_size = 0.5  # Example value, adjust as needed
reverberation_time = 1.0  # Example value, adjust as needed
audio_reverberated = apply_reverberation(audio, fs, room_size, reverberation_time)


# Apply the stretch effect
stretch_ratio = 1.2  # Example value, adjust as needed
audio_stretched = stretch(audio, stretch_ratio)

# Apply the shift effect
shift_sec = 0.5  # Example value, adjust as needed
audio_shifted = shift(audio, shift_sec, fs)

# Apply the white noise effect
noise_var_coeff = 0.1  # Example value, adjust as needed
audio_with_white_noise = add_white_noise(audio, noise_var_coeff)

save_dir = "augmented_audios/"
# Save the augmented audio data to new files
wavfile.write(save_dir + "echo_augmented_" + file_name, fs, audio_with_echo)
wavfile.write(save_dir + "pitch_shift_augmented_" + file_name, fs, audio_with_pitch_shift)
wavfile.write(save_dir + "reverberation_augmented_" + file_name, fs, audio_reverberated)
wavfile.write(save_dir + "stretch_augmented_" + file_name, fs, audio_stretched)
wavfile.write(save_dir + "shift_augmented_" + file_name, fs, audio_shifted)
wavfile.write(save_dir + "white_noise_augmented_" + file_name, fs, audio_with_white_noise)
# Save a copy of the original audio file
wavfile.write(save_dir + "original_copy.wav", fs, audio)
