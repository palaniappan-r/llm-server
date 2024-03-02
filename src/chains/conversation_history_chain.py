from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from memory.ecm import ExtendedConversationBufferMemory

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class ConversationCharacter:
    def __init__(self , char_pers , char_name , place , activity , local_llm):
        self.char_name = char_name
        self.char_pers = char_pers
        self.place = place
        self.activity = activity
        self.local_llm = local_llm

        self.char_prompt , self.char_instr = self.generate_char_prompt()


    def initialize_char_prompt(self):
        p_template = f""" {B_INST}
                        {B_SYS}
                        {self.char_prompt}
                        ---

                        {E_SYS}
                        ---
                        Instructions:
                        {self.char_instr}
                        ---

                        {{history}}
                        Human: {{input}}
                        {self.char_name}: 
                        {E_INST}
                    """
        char_prompt = PromptTemplate(
            input_variables=["history","input","char_action"], template=p_template
        )
        
        return char_prompt

    def generate_char_prompt(self):
        char_prompt = f"""
        Your name is {self.char_name}.
        """
        
        char_instr = f"""
        You will have a conversation with a User, and you will engage in a dialogue with them.

        These are the list of actions you are supposed to perform:
        {{char_action}}

        Engage in conversation with the user
        """

        return char_prompt , char_instr

    def create_chain(self):
        # init conv chain
        conversation_with_history = ConversationChain(
            llm = self.local_llm, 
            prompt = self.initialize_char_prompt(),
            memory = ExtendedConversationBufferMemory(extra_variables=["char_action"] , human_prefix = "Human" , ai_prefix = self.char_name),
            verbose = True
        )
        
        return conversation_with_history