import transformers
import torch
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-bs', '--beam-size', type=int, default=4)
parser.add_argument('-mn', '--model-name', required=True)
parser.add_argument('-lp', '--length-penalty', default=1.0, type=float)
parser.add_argument('-tn', '--tokenizer-name')
parser.add_argument('--max-sentences', type=int, default=16)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--do-sample', action='store_true')
parser.add_argument('--top-k', type=int)
parser.add_argument('--top-p', type=float)
parser.add_argument('--max-length-a', type=float, default=1.5)
parser.add_argument('--max-length-b', type=int, default=50)



if __name__ == "__main__":
    args = parser.parse_args()
    args = parser.parse_args()
    device = torch.device("cuda")

    model_name = args.model_name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name).to(device)
    model.config.length_penalty = args.length_penalty
    if model.config.pad_token_id == tokenizer.vocab['<extra_id_1>']:
        setattr(model.config, 'pad_token_id', model.config.decoder_start_token_id)
    model.eval()
    finished = False
    bar = tqdm.tqdm()
    with open(args.input, encoding='utf8') as fp, \
         open(args.output, 'w', encoding='utf8') as ofp:
        while not finished:
            lines = []

            for _ in range(args.max_sentences):
                line = fp.readline()
                if line == "":
                    finished = True
                    break
                lines.append(line.strip())
                bar.update()
            
            inputs = tokenizer(lines, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(device)
            if getattr(model.config, 'n_positions', None):
                max_length = model.config.n_positions
            elif getattr(model.config, 'max_position_embeddings', None):
                max_length = model.config.max_position_embeddings
            else:
                max_length = 256
            max_length =  min(max_length, int(args.max_length_a * input_ids.shape[-1]) + args.max_length_b)

            if args.do_sample:
                setattr(args, 'beam_size', 1)

            outputs = model.generate(
                input_ids,
                do_sample=args.do_sample,
                num_beams=args.beam_size,
                num_return_sequences=1,
                max_length=max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                length_penalty=args.length_penalty
            )

            for i in range(outputs.shape[0]):
                ofp.write(tokenizer.decode(outputs[i], skip_special_tokens=True) + '\n')
            ofp.flush()
