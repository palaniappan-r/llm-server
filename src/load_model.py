from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class LoadModel:
    def __init__(self , memory_config , w1 , w2):
        self.w1 = w1
        self.w2 = w2
        self.memory_config = memory_config
        self.model , self.tokenizer = self.load_model()

    def load_model(self):   
        model_name_or_path = self.w1
        model_basename = self.w2

        use_triton = False
        max_memory = self.memory_config
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True , trust_remote_code=True)

        model = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path,
                model_basename=model_basename,
                inject_fused_attention=False,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None,
                wbits = -1,
                max_memory = max_memory
                )

        return model,tokenizer
    
    def load_llm(self ,context_size=512*8 ,temp=0 , rp=1 , tp=1  ,tk=0):
        llm_pipeline = pipeline(
        "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1
        )
        
        local_llm = HuggingFacePipeline(pipeline=llm_pipeline)
        
        return local_llm