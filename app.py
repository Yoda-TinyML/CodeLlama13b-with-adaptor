from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class InferlessPythonModel:
    def initialize(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        #adapter_file = "G-ML-Hyly/stg-cli13b-t6-cdp-ca.mt.him.cln.inter-b4s1e1-20231220-1052"
        config = PeftConfig.from_pretrained("G-ML-Hyly/stg-cli13b-t6-cdp-ca.mt.him.cln.inter-b4s1e1-20231219-1611")
        #config = PeftConfig.from_pretrained(adapter_file)
        self.model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
        self.model = PeftModel.from_pretrained(self.model, "G-ML-Hyly/stg-cli13b-t6-cdp-ca.mt.him.cln.inter-b4s1e1-20231219-1611",quantization_config=quantization_config)
        #self.model = PeftModel.from_pretrained(self.model, adapter_file)
        self.model.to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
    
    def infer(self, inputs):
        prompt = inputs["prompt"]
        batch = self.tokenizer(prompt, return_tensors='pt').to("cuda:0")
        output_tokens = self.model.generate(**batch, max_new_tokens=50)
        output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return {"generated_output": output}

    def finalize(self,args):
        self.model = None
        self.tokenizer = None
