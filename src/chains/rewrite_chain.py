import re
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class Rewrite_Chain:
    def __init__(self, char_name , local_llm):
        self.char_name = char_name
        self.local_llm = local_llm
        self.rewrite_prompt = self.initialize_rewrite_prompt()
        # self.rewrite_chain = None
        # self.create_rewrite_chain()

    def initialize_rewrite_prompt(self):
        p_template = f""" {B_INST}
                        {B_SYS} 
                        Consider the following piece of text, written by {self.char_name}
                        
                        Text:
                        {{input}}

                        Emotion:
                        {{emotion}}
                        
                        {E_SYS}
                        ---
                        Rewrite with the provided Emotion. Do change the meaning of the given text.

                        The rewritten response should be in lowercase within 20 words.
                        Lable the rewritten response with "NEW_RESPONSE:"
                         
                        NEW_RESPONSE:{E_INST}
                    """
        char_prompt = PromptTemplate(
            input_variables=["emotion","input"], template=p_template
        )
        
        return char_prompt


    def create_rewrite_chain(self):
        self.rewrite_chain = ConversationChain(
            llm = self.local_llm, 
            prompt = self.initialize_rewrite_prompt(),
            verbose = False
        )

    def invoke(self , m_input , m_emotion):
        output = self.local_llm(self.rewrite_prompt.format(   
            input = m_input,
            emotion = m_emotion
        ))

        return output
