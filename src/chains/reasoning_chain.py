from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class ReasoningChain:
    def __init__(self, char_name , local_llm):
        self.char_name = char_name
        self.local_llm = local_llm
        self.coa_prompt = self.initialize_coa_prompt()
        self.reasoning_chain = None
        self.create_reasoning_chain()

    def initialize_reasoning_prompt(self):
        p_template = f""" {B_INST}
                        {B_SYS} 
                        You are an AI bot built to give other {self.char_name} a course of actions, to help them answer the user's query. 
                        {{input}}
                        
                        {E_SYS}
                        Your task is to generate a course of actions to follow, based on the user's query.
                        
                        User Query:
                        {E_INST}
                    """
        char_prompt = PromptTemplate(
            input_variables=["input"], template=p_template
        )
        
        return char_prompt


    def create_reasoning_chain(self):
        self.reasoning_chain = ReasoningChain(
            llm = self.local_llm, 
            prompt = self.initialize_coa_prompt(),
            verbose = False
        )

    def invoke(self, u_input):
        output = self.local_llm(self.coa_prompt.format(   
            input = u_input
        ))

        return output

