# **Snapy**
<a href="https://github.com/aymenbrahimdjelloul/Snapy/releases/latest">
    <img src="https://img.shields.io/github/v/release/aymenbrahimdjelloul/Snapy?color=green&label=Download&style=for-the-badge" alt="Download Latest Release">
</a>


**Snapy** is a powerful and lightweight AI Chatbot It’s designed to enhance your productivity by offering intelligent with Command-line interface to test it

~~~bash
> Hello, Snapy!
🤖 Snapy: Hi there! How can I assist you today?
~~~

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

## **Incoming Updates ?**

- **Deploy Snapy in Web application**
- **Improve and implement conversation context handling**
- **Improve Online search engine**
- **Update Snapy Dataset with more intents**
---

## **Why Snapy ?**

- **Terminal-Native:** Runs entirely in the command
- **Fast & Lightweight:** Built with speed and simplicity in mind  
- **AI-Enhanced:** Use local or online models to power intelligent chat  
- **Use Online searching for geting informations
---

## **Getting Started**

~~~bash
git clone https://github.com/aymenbrahimdjelloul/Snapy.git
cd Snapy
python snapy.py
~~~

## **Contribute**

We welcome contributions to **HashRipper**! Whether you're fixing a bug, suggesting a feature, or submitting code, your help makes this tool better.

**To contribute:**
1. Fork the repository.
2. Create a new branch (`feature/my-feature` or `fix/my-bug`).
3. Make your changes and test thoroughly.
4. Submit a pull request with a detailed description.

---

## **Thanks**

Special thanks to:

- **Cybersecurity Experts** – for providing best practices and guidance.
- **Open Source Contributors** – for generously sharing your time and skills.
  
Thank you for using **HashRipper** – Stay safe and crack responsibly!

---

### License : 

~~~
MIT License

Copyright (c) 2025 Aymen Brahim Djelloul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

~~~
