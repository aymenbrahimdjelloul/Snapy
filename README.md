# **Snapy**

#### A Terminal-Based AI Chatbot Assistant

**Snapy** is a powerful and lightweight AI Chatbot It’s designed to enhance your productivity by offering intelligent command handling, styled outputs, typing effects, developer tools, and persistent chat history—all from your terminal.

---

## **Key Features**

- ✅ **Command Handling:** Interprets and executes structured commands  
- ✅ **Typing Effects:** Simulates human-like typing for immersive responses  
- ✅ **Styled Headers & UI:** Clean, readable terminal interface  
- ✅ **Persistent Chat History:** Maintains context between sessions  
- ✅ **Developer Mode:** Enables detailed logging and debug insights  
- ✅ **Extensible Design:** Easily integrates new commands or modules  

---

## **Usage**

~~~python
# Import Snapy
from snapy import Snapy

# Create Snapy object
snapy = Snapy()

# Get some responses with random user input
greeting_response = snapy.generate_response("Hello there !")
question_response1 = snapy.generate_response("who are you ?")
qeustion_response2 = snapy.generate_response("What is Festina ?")

# Print results
print(greeting_response)
print(question_response1)
print(question_response2)

~~~

## **How It Works**

Snapy processes user input using a custom parser that distinguishes between chat prompts and system-level commands. Key components include:

- **Command Router:** Routes structured inputs to their respective handlers  
- **Renderer:** Applies styles and visual effects for clean terminal output  
- **Session Manager:** Saves chat history across sessions  
- **Developer Tools:** Offers insights like execution time, debug logs, and internal state if enabled

Snapy can be customized for your workflows—whether you're automating tasks, searching local files, or interacting with APIs.

---

## **Why Snapy?**

- ** Terminal-Native:** Runs entirely in the command
- ** Fast & Lightweight:** Built with speed and simplicity in mind  
- ** AI-Enhanced:** Use local or online models to power intelligent chat  
- ** Use Online searching for geting informations
---

## **Getting Started**

```bash
git clone https://github.com/aymenbrahimdjelloul/Snapy.git
cd Snapy
python snapy.py
