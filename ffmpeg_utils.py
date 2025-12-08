import os
import re
import shutil
import platform
import subprocess as sp
from platform import system


def is_wsl():
    """Check if running in WSL environment."""
    try:
        return 'microsoft' in platform.release().lower() or 'WSL_DISTRO_NAME' in os.environ
    except Exception:
        return False


def get_ffmpeg_path():
    """
    Find ffmpeg executable path.
    Checks environment variable, system PATH, and common installation locations.
    Returns None if ffmpeg is not found.
    """
    env_binary = os.getenv("FFMPEG_BINARY")
    if env_binary and os.path.isfile(env_binary):
        print(f"✅ Using ffmpeg from FFMPEG_BINARY: {env_binary}")
        return env_binary

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe

    windows_candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    wsl_candidates = [
        "/mnt/c/ffmpeg/bin/ffmpeg.exe",
        "/mnt/c/Program Files/ffmpeg/bin/ffmpeg.exe",
        "/mnt/c/Program Files (x86)/ffmpeg/bin/ffmpeg.exe",
    ]
    linux_candidates = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]

    if system() == "Windows":
        for candidate in windows_candidates:
            if os.path.isfile(candidate):
                print(f"✅ Using ffmpeg for Windows: {candidate}")
                return candidate

    if is_wsl():
        for candidate in wsl_candidates:
            if os.path.isfile(candidate):
                print(f"✅ Using Windows ffmpeg via WSL: {candidate}")
                return candidate

    for candidate in linux_candidates:
        if os.path.isfile(candidate):
            print(f"✅ Using ffmpeg for Linux: {candidate}")
            return candidate

    print("❌ FFMPEG binary not found. Please install ffmpeg or set FFMPEG_BINARY.")
    return None


def adapt_path_for_ffmpeg(path, ffmpeg_exe):
    """
    Adapt file path for ffmpeg execution.
    Converts WSL paths (/mnt/c/...) to Windows paths (C:\\...) when using Windows ffmpeg.
    """
    if not path or not ffmpeg_exe:
        return path

    # Windows executables can't consume /mnt/* paths, convert if needed
    if ffmpeg_exe.lower().endswith(".exe"):
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)", path)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2).replace('/', '\\')
            return f"{drive}:\\{rest}"
    return path


def copy_audio_to_video(input_video_path, temp_video_path, output_path):
    """
    Copy audio from original video to processed video using ffmpeg.
    Re-encodes video to match input quality and codec.

    Args:
        input_video_path: Path to original video with audio
        temp_video_path: Path to processed video without audio
        output_path: Path for final output video with audio

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"[copy_audio_to_video] Called with:")
    print(f"  - input_video_path: {input_video_path}")
    print(f"  - temp_video_path: {temp_video_path}")
    print(f"  - output_path: {output_path}")

    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe:
        error_msg = "FFMPEG not found. Cannot merge audio."
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    # Ensure the existence of output directory
    output_dir = os.path.dirname(output_path)
    print(f"[copy_audio_to_video] Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Check if temp video exists
    print(f"[copy_audio_to_video] Checking temp video existence...")
    if not os.path.isfile(temp_video_path):
        error_msg = f"Temp video file not found: {temp_video_path}"
        print(f"❌ {error_msg}")
        # List files in the directory to help debug
        temp_dir = os.path.dirname(temp_video_path)
        if os.path.exists(temp_dir):
            print(f"Files in {temp_dir}:")
            for f in os.listdir(temp_dir):
                print(f"  - {f}")
        raise FileNotFoundError(error_msg)

    print(f"✓ Temp video exists: {temp_video_path}")

    if not os.path.isfile(input_video_path):
        error_msg = f"Original input file not found: {input_video_path}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)

    print(f"✓ Input video exists: {input_video_path}")

    # Ensure output has .mp4 extension
    if not output_path.endswith('.mp4'):
        output_path = os.path.splitext(output_path)[0] + '.mp4'

    # Temporary output file
    temp_output_with_audio = output_path.replace(".mp4", "_with_audio.mp4")

    # Use high-quality H.264 encoding to match or exceed input quality
    command = [
        ffmpeg_exe, "-y",
        "-i", adapt_path_for_ffmpeg(temp_video_path, ffmpeg_exe),  # Edited video without audio
        "-i", adapt_path_for_ffmpeg(input_video_path, ffmpeg_exe),  # Original video with audio
        "-map", "0:v",  # Video from the processed file
        "-map", "1:a?",  # Audio from original video (optional)
        # Video encoding settings - high quality H.264
        "-c:v", "libx264",
        "-preset", "slow",  # Slower = better quality
        "-crf", "18",  # CRF 18 = visually lossless (0-51, lower = better quality)
        "-pix_fmt", "yuv420p",  # Compatibility
        # Audio encoding settings - copy or re-encode with high quality
        "-c:a", "aac",
        "-b:a", "192k",  # High quality audio bitrate
        "-shortest",
        adapt_path_for_ffmpeg(temp_output_with_audio, ffmpeg_exe)
    ]

    print(f"[copy_audio_to_video] Running FFmpeg command...")
    print(f"[copy_audio_to_video] Command: {' '.join(command)}")

    result = sp.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = f"FFmpeg failed with return code {result.returncode}"
        print(f"❌ {error_msg}")
        print("STDERR:")
        print(result.stderr)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        raise RuntimeError(f"{error_msg}\n{result.stderr}")

    print(f"✓ FFmpeg completed successfully")
    print(f"[copy_audio_to_video] Checking if temp output was created...")

    # Check if temp output file was created
    if not os.path.exists(temp_output_with_audio):
        error_msg = f"FFmpeg did not create output file: {temp_output_with_audio}"
        print(f"❌ {error_msg}")
        print(f"Files in output directory:")
        output_dir = os.path.dirname(temp_output_with_audio)
        for f in os.listdir(output_dir):
            print(f"  - {f}")
        raise FileNotFoundError(error_msg)

    print(f"✓ Temp output exists: {temp_output_with_audio}")

    # Replace old output with audio output
    print(f"[copy_audio_to_video] Moving {temp_output_with_audio} → {output_path}")
    shutil.move(temp_output_with_audio, output_path)

    # Verify final file exists
    if not os.path.exists(output_path):
        error_msg = f"Final output file not found after move: {output_path}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)

    file_size = os.path.getsize(output_path)
    print(f"✅ Video re-encoded with high quality and audio merged: {output_path}")
    print(f"✅ Final file size: {file_size / (1024*1024):.2f} MB")

    # Clean up temp .avi file
    if temp_video_path.endswith('.avi') and os.path.exists(temp_video_path):
        print(f"[copy_audio_to_video] Cleaning up temp .avi file: {temp_video_path}")
        try:
            os.remove(temp_video_path)
            print(f"✓ Temp .avi file removed")
        except Exception as e:
            print(f"Warning: Could not remove temp file: {e}")

    return True
