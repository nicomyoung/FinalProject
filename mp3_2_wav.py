import os
import pydub

ffmpeg_path = "C:/Final Project/fftools/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        audio = pydub.AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        print(f"Successfully converted {mp3_path} to {wav_path}")
    except Exception as e:
        print(f"Failed to convert {mp3_path} due to {e}")

def main(start_index=0):
    source_directory = "C:/Final Project/localAudios/mp3Files"
    target_directory = "C:/Final Project/localAudios/wavFiles"

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    all_files = [f for f in os.listdir(source_directory) if f.endswith(".mp3")]

    # Resume from the last index
    for i in range(start_index, len(all_files)):
        filename = all_files[i]
        source_file_path = os.path.join(source_directory, filename)
        base_filename = os.path.splitext(filename)[0]
        target_file_path = os.path.join(target_directory, base_filename + ".wav")
        convert_mp3_to_wav(source_file_path, target_file_path)

    print("Conversion completed!")

if __name__ == "__main__":
    main(start_index=631)
