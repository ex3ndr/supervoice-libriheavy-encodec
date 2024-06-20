from encodec import EncodecModel
from encodec.utils import convert_audio
from audio import load_mono_audio
from pathlib import Path

def get_model():
    if not hasattr(get_model, "model"):
        get_model.model = model = EncodecModel.encodec_model_24khz()
        get_model.model.set_target_bandwidth(6.0)
    return get_model.model

def encode_parallel(args):
    process_id = multiprocessing.current_process()._identity[0]
    files, output_dir, index = args
    file = files[index]
    device = "cpu" if alignment else "cuda:" + str(process_id % torch.cuda.device_count())
    model = get_model()

    # Load and preprocess
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    # Encode
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

    # Save
    torch.save(codes, os.path.join(output_dir, file.replace(".wav", ".pt")))

def execute_run():

    # Indexing files
    print("Build file index...")
    files = Path("./external_datasets/librilight/").rglob("*.wav")

    # Prepare directory
    prepared_dir = Path("./encoded_datasets/librilight/")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    # Process all files    
    with multiprocessing.Manager() as manager:
        files = manager.list(files)
        args_list = [(files, prepared_dir, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=32) as pool:
            for result in tqdm(pool.imap_unordered(encode_parallel, args_list, chunksize=32), total=len(files)):
                pass
    

if __name__ == "__main__":
    execute_run()