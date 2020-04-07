import os
import time
import re
from slackclient import SlackClient

import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_filter import WordFilter, StopwordFilter
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PorterStemmer
from allennlp.data.tokenizers.token import Token
import allennlp.data.dataset_readers.semantic_dependency_parsing as sdp
from allennlp.predictors.predictor import Predictor


# instantiate Slack client
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
# starterbot's user ID in Slack: value is assigned after the bot starts up
starterbot_id = None

# constants
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

sid = SentimentIntensityAnalyzer()
ps = PorterStemmer()

food_list = []
for line in open('food.txt'):
  food = line.rstrip('\n')
  food_list.append(food)
  food_list.append(food+"s")
  
predictor = Predictor.from_path("srl-model-2018.05.25.tar.gz")
dependency_predictor = Predictor.from_path("biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
items = []
location = ""

class FoodItem:
  def __init__(self, name, quantity, attributes):
    self.name = name
    self.quantity = quantity
    self.attributes = attributes
  def to_string(self):
    string = str(self.quantity)
    for a in self.attributes:
      string += " "
      string += a
    for n in self.name:
      string += " "
      string += n
    return string
   

def filter_for_food(string):
  global food_list, dependency_predictor, items
  result = dependency_predictor.predict(sentence=string)
  words = result['words']
  part_of_speech = result ['pos']
  attributes = []
  name = []
  for index in range(len(words)):
    pos = part_of_speech[index]
    if pos == "DT":
      if words[index] == "a" or words[index] == "an":
        quantity = 1
    if pos == "CD":
      quantity = words[index]
    if pos == "JJ" or pos == "JJR" or pos == "JJS":
      attributes.append(words[index])
    if pos == "NN" or pos == "NNS" or pos == "NNP" or pos == "NNPS":
      name.append(words[index])
    if pos == "," or pos == "." or pos == "CC":
      if len(name) != 0:
        for n in name:
          if n in food_list:
            item = FoodItem(name,quantity,attributes)
            items.append(item)
            attributes = []
            name = []
            break
  for n in name:
    if n in food_list:
      item = FoodItem(name,quantity,attributes)
      items.append(item)
      attributes = []
      name = []
      break
    
def process(command):
  global partial_response, ss, sid, predictor, items, location
  ss = sid.polarity_scores(command)
  response = ""
  if partial_response == False:
      result=predictor.predict(command)
      for dictionary in result['verbs']:
        verb = dictionary['verb']
        token = Token(text=verb)
        token=ps.stem_word(token)
        if token.text == 'order':
          if ss['compound'] >= 0.0 and ss['compound'] <= 0.5 and ss['neu'] > ss['pos']:
            try:
              response = dictionary['description']
              try:
                arg1=response.split('ARG1: ')[1].split(']')[0]
                if 'from' in arg1:
                  location = arg1.split('from')[1]
                  filter_for_food(arg1.split('from')[0])
                else:
                  filter_for_food(arg1)
              except:
                print("No arg1")

              try:
                arg2=response.split('ARG2: ')[1].split(']')[0]
                if 'from' in arg2:
                  location = arg2.split('from')[1]
              except:
                print("No arg2")

              partial_order = ""
              for item in items:
                partial_order += " "
                partial_order += item.to_string()
                partial_order += ","
                
              if location != "":
                response="Would you like me to order: "+partial_order.rstrip(",")+" from "+location+"?\n<Yes/No/I also want to order...>"
              else:
                response="Would you like me to order: "+partial_order.rstrip(",")+" from LOCATION NOT SET?\n<Yes/No/I also want to order...>"
              partial_response = True
            except:
              print("We did an oopsie here")
          else:
            response="If you want me to order some food, try: @Starter Bot I want to order <<food>>"
  else:
      if "Yes" in command:
        order = ""
        for item in items:
            order += " "
            order += item.to_string()
            order += ","
        if location != "":
                response="I will order: "+order.rstrip(",")+" from "+location
                partial_response = False
                location=""
                items=[]
        else:
            response="Please set the location from which to order by: I want to order from <location>"
      elif "No" in command:
        response="Order canceled"
        partial_response = False
        location = ""
        items=[]
      else:
        result=predictor.predict(command)
        for dictionary in result['verbs']:
          verb = dictionary['verb']
          token = Token(text=verb)
          token=ps.stem_word(token)
          if token.text == 'order':
            if ss['compound'] >= 0.0 and ss['compound'] <= 0.5 and ss['neu'] > ss['pos']:
              try:
                response = dictionary['description']
                try:
                  arg1=response.split('ARG1: ')[1].split(']')[0]
                  if 'from' in arg1:
                    location = arg1.split('from')[1]
                    filter_for_food(arg1.split('from')[0])
                  else:
                    filter_for_food(arg1)
                except:
                  print("No arg1")

                try:
                    arg2=response.split('ARG2: ')[1].split(']')[0]
                    if 'from' in arg2:
                        location = arg2.split('from')[1]
                except:
                    print("No arg2")

                partial_order = ""
                for item in items:
                  partial_order += " "
                  partial_order += item.to_string()
                  partial_order += ","

                if location != "":
                  response="Would you like me to order: "+partial_order.rstrip(",")+" from "+location+"?\n<Yes/No/I also want to order...>"
                else:
                  response="Would you like me to order: "+partial_order.rstrip(",")+" from LOCATION NOT SET?\n<Yes/No/I also want to order...>"
              except:
                print("We did an oopsie here")
            else:
              response="If you want me to order some more, try: @Starter Bot I want to order <<food>>"
  return response

def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            user_id, message = parse_direct_mention(event["text"])
            if user_id == starterbot_id:
                return message, event["channel"]
    return None, None

def parse_direct_mention(message_text):
    """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. If there is no direct mention, returns None
    """
    matches = re.search(MENTION_REGEX, message_text)
    # the first group contains the username, the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

def handle_command(command, channel):
    """
        Executes bot command if the command is known
    """
    # Default response is help text for the user
    default_response = "Not sure what you mean"
    default_food_response = "I didn't quite catch that, but I see that you mentioned something about food. If you want me to order some food, try: @Starter Bot Order <<food>>"

    # Finds and executes the given command, filling in response
    # This is where you start to implement more commands!
    response = None
    	
    response=process(command)

    if response == None:
        for word in command:
            if word in food_list:
                response=default_food_response
                break

    # Sends the response back to the channel
    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text=response or default_response
    )

if __name__ == "__main__":
    global partial_response
    if slack_client.rtm_connect(with_team_state=False):
        print("Starter Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        starterbot_id = slack_client.api_call("auth.test")["user_id"]
        partial_response = False
        while True:
            command, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                handle_command(command, channel)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")
