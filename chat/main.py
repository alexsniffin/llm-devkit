import argparse
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class Chatbot:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, torch_dtype=torch.bfloat16)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def get_response(self, prompt):
        inputs = self.tokenizer.encode_plus(prompt, return_tensors="pt", padding='max_length', max_length=100)
        if next(self.model.parameters()).is_cuda:
            inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
        start_time = time.time()
        tokens = self.model.generate(input_ids=inputs['input_ids'],
                                     attention_mask=inputs['attention_mask'],
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     max_new_tokens=400)
        end_time = time.time()
        output_tokens = tokens[0][inputs['input_ids'].shape[-1]:]
        output = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        time_taken = end_time - start_time
        return output, time_taken


def main(model):
    chatbot = Chatbot("/models/" + model)
    while True:
        user_input = input("Enter your prompt: ")
        if user_input.lower() == 'quit':
            break
        output, time_taken = chatbot.get_response(user_input)
        print("\033[33m" + output + "\033[0m")
        print("Time taken to process: ", time_taken, "seconds")
    print("Exited the program.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('model', type=str, help='model name')

    args = parser.parse_args()

    main(args.model)
