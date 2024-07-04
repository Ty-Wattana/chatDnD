from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

# llm init
llm = Ollama(
    # model="gemma2:27b"
    model="llama3"
)


system_message = """
You are now the guide of a mystical journey in the Whispering Woods. 
A traveler named Elara seeks the lost Gem of Serenity. 
You must navigate her through challenges, choices, and consequences, 
dynamically adapting the tale based on the traveler's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining Elara's fate. 

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "THE END" in all capital letters, I will search for this text to end the game
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ]
)

chain = prompt | llm

# simple memory system

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="human_input",
    history_messages_key="chat_history",
)

chat_message_history = SQLChatMessageHistory(session_id="chat_hist_id", connection_string="sqlite:///sqlite.db")
chat_message_history.clear()

config = {"configurable": {"session_id": "chat_hist_id"}}
choice = "start"

while True:
    response = chain_with_history.invoke({"human_input": choice},config=config)
    print(response.strip())

    if "THE END" in response:
        chat_message_history.clear()
        break

    choice = input("Your reply: ")