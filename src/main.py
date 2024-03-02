import os
import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse

from load_model import LoadModel
from chains.conversation_history_chain import ConversationCharacter
from chains.reasoning_chain import ReasoningChain
from chains.rewrite_chain import Rewrite_Chain

chainDict = {}
historyDict = {}

class LocalData(object):
    records = {}


class HTTPRequestHandler(BaseHTTPRequestHandler):
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(HTTPRequestHandler, self).end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
        
    def do_POST(self):
        url = self.path
        path = url.split('?')[0]
        
        # Histroy endpoint to keep track of conversations
        if path == '/history':
            self.send_response(200, 'Ok')
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            conv_id = query_params.get('conv_id', [None])[0]
            
            self.send_response(200, 'Ok')
            self.end_headers()
            response = json.dumps({
                'conversation': historyDict[conv_id]
            })
            
            self.wfile.write(response.encode('utf-8'))
        
        # A very basic login system to differentiate between users
        elif path == '/login':
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length).decode('utf-8'))
            if body["username"] in passdict.keys():
                if passdict[body["username"]] == body["password"]:
                    self.send_response(200, 'Ok')
                    self.end_headers()
                    response = json.dumps({
                        'result': "User validated"
                    })
                else:
                    self.send_response(401, 'Unauthorized')
                    self.end_headers()
                    response = json.dumps({
                        'result': "Wrong Password"
                    })
            else:
                self.send_response(401, 'Unauthorized')
                self.end_headers()
                response = json.dumps({
                    'result': "User does not exist"
                })
            self.wfile.write(response.encode('utf-8'))
        
        #Endpoint to keep track of currently active chains
        elif path == '/active':
            self.send_response(200, 'Ok')
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            conv_id = query_params.get('conv_id', [None])[0]
            
            self.send_response(200, 'Ok')
            self.end_headers()
            response = json.dumps({
                'activeChains': len(chainDict)
            })
            self.wfile.write(response.encode('utf-8'))
        
        #Endpoint to interact with each chain
        elif path == '/':
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            conv_id = query_params.get('conv_id', [None])[0]
            
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length).decode('utf-8'))
            user_input = body["u_input"]
            
            print("\nPrompting Chain :" , conv_id)
            
            actions = reasoning_chain.invoke(user_input)
            ai_response = chainDict[conv_id].run({"input":user_input , "char_action":actions})

            if(f"{char_name}:" in ai_response):
                ai_response = ai_response.split(f"{char_name}:")[1]
            
            if("response:" in ai_response):
                ai_response = ai_response.split("response:")[1]
            
            ai_response = re.sub(r'\*.*?\*', '', ai_response).strip()

            print(f"\n Response : {ai_response} \n")
            
            historyDict[conv_id].append([user_input , ai_response])
   
            self.send_response(200, 'Ok')
            self.end_headers()
            response = json.dumps({
                'conversation': historyDict[conv_id]
            })
            
            self.wfile.write(response.encode('utf-8'))
            
        else:
            self.send_response(403)

    def do_GET(self):    
        url = self.path

        path = url.split('?')[0]
        print(url)
        print(path)
        
        if path == '/':
            self.send_response(200, 'Ok')
            self.end_headers()
            response = json.dumps({
                'result': "API running"
            })
            
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            conv_id = query_params.get('conv_id', [None])[0]
            
            if conv_id not in chainDict.keys():
                chainDict[conv_id] = bot.create_chain()
            if conv_id not in historyDict.keys():
                historyDict[conv_id] = []
            
            print("\nCreated Chain : " , conv_id)
            self.wfile.write(response.encode('utf-8'))
                
        else:
            self.send_response(403)
        self.end_headers()

        
if __name__ == '__main__':
    
    local_llm = LoadModel(
        memory_config = {0: "25GIB", 1: "25GIB"} ,
        w1 = "./models/TheBloke_Llama-2-13B-chat-GPTQ" , 
        w2 = "gptq_model-4bit-128g" 
        ).load_llm()

    char_name = "Bot"
    
    char_pers = """
    Bot is a an AI assistant, designed to help users with their queries and problems. He is a friendly and helpful AI, and is always ready to help users with their problems.
    """
    
    char_location = """
    """
    
    char_activity = """
    Bot is currently helping a user with their query.
    """

    bot = ConversationCharacter(char_pers , char_name , char_location , char_activity , local_llm)
    reasoning_chain = ReasoningChain(char_name , local_llm)

    convDict = {}
    historyDict = {}

    passdict = {}

    f = open("./userlist.txt")
    for i in f.readlines():
        passdict[i.split(":")[0]] = i.split(":")[1].strip()

    server = ThreadingHTTPServer(('localhost', 8000), HTTPRequestHandler)
    os.system('clear')
    print('Starting server on port 8000\n')
    
    chainDict['1'] = bot.create_chain()
    historyDict['1'] = []
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    
    server.server_close()
    logging.info('Stopping httpd...\n')
    
    
