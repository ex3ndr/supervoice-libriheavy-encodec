# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Imports
import gzip
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import os
import json
import math

# Process single file
def get_model():
    if not hasattr(get_model, "model"):
        process_id = multiprocessing.current_process()._identity[0]
        device = "cuda:" + str(process_id % torch.cuda.device_count())
        get_model.device = device
        get_model.model = model = EncodecModel.encodec_model_24khz()
        get_model.model.set_target_bandwidth(6.0)
        get_model.model.to(get_model.device)
    return get_model.device, get_model.model

def clean_text(s: str) -> str:
    table = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")
    s = s.translate(table)
    return s.strip()

def encode_parallel(args):
    files, output_dir, index = args
    file = files[index]['path']
    cuts = files[index]['cuts']
    device, model = get_model()

    # Load sourceaudio
    source, sr = torchaudio.load(file)

    # Process cuts
    for cut in cuts:
        id, start, duration, text = cut
        wav = source[:, int(start * sr):int((start + duration) * sr)]
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0)

        # Encode
        with torch.no_grad():
            encoded_frames = model.encode(wav.to(device))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze().cpu()

        # Save codecs
        output_file = Path(output_dir) / Path(id + ".pt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():        
            print("File exists", output_file)
        torch.save(codes, output_file)
        
        # Save text
        output_file = Path(output_dir) / Path(id + ".txt")
        if output_file.exists():        
            print("File exists", output_file)
        with open(output_file, "w") as f:
            f.write(text)

def execute_run():

    # Small
    # index_path = "./external_datasets/libriheavy/libriheavy_cuts_small.jsonl.gz"
    # files_path = "./external_datasets/librilight/"
    # output_path = "./encoded_datasets/librilight/"

    # Medium
    # index_path = "./external_datasets/libriheavy/libriheavy_cuts_medium.jsonl.gz"
    # files_path = "./external_datasets/librilight-medium/"
    # output_path = "./encoded_datasets/librilight-medium/"

    # Large
    index_path = "./external_datasets/libriheavy/libriheavy_cuts_large.jsonl.gz"
    files_path = "./external_datasets/librilight-large/"
    output_path = "./encoded_datasets/librilight-large/"

    # Indexing files
    print("Build file index...")
    files = []
    files_map = {}
    existing_id = {}
    with gzip.open(index_path, "r") as f:
        for line in f:
            cut = json.loads(line)
            start = math.floor(1000 * cut["start"]) / 1000
            duration = math.floor(1000 * cut["duration"]) / 1000

            # Load audio
            wav_id = cut["recording"]["id"]
            id = cut["supervisions"][0]["id"]
            if wav_id.startswith("small/"):
                wav_id = wav_id[len("small/"):]
            if wav_id.startswith("medium/"):
                wav_id = wav_id[len("medium/"):]
            if wav_id.startswith("large/"):
                wav_id = wav_id[len("large/"):]
            if id.startswith("small/"):
                id = id[len("small/"):]
            if id.startswith("medium/"):
                id = id[len("medium/"):]
            if id.startswith("large/"):
                id = id[len("large/"):]

            # Check ID
            if id in existing_id:
                print("ID exists", id)
            existing_id[id] = True

            # Check if exists
            if (Path(output_path) / Path(id + ".pt")).exists():
                continue

            # Load text
            text = cut["supervisions"][0]["custom"]["texts"][0]
            text = clean_text(text)
        
            # Find index
            if wav_id not in files_map:
                files_map[wav_id] = len(files)
                files.append({ "path": files_path + wav_id + ".flac", "cuts": []})
            index = files_map[wav_id]

            # Append
            files[index]['cuts'].append((id, start, duration, text))

    # Prepare directory
    prepared_dir = Path(output_path)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    # Process all files    
    print("Processing...")
    with multiprocessing.Manager() as manager:
        files = manager.list(files)
        args_list = [(files, prepared_dir, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=2) as pool:
            for result in tqdm(pool.imap_unordered(encode_parallel, args_list, chunksize=1), total=len(files)):
                pass
    

if __name__ == "__main__":
    execute_run()