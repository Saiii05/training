AI Intern Assignment: Object Detection Model - Experience Report
This report details my personal experience completing the Object Detection Model assignment.

1. Challenges Faced During Implementation
Embarking on the object detection project was both exciting and daunting. One of the primary challenges I encountered was understanding the intricacies of different object detection algorithms, such as YOLO, SSD, and Faster R-CNN. Each had its own architecture and nuances, making it challenging to decide which one best suited the project's requirements.

Another significant hurdle was data preprocessing. Ensuring that the dataset was correctly annotated and formatted for training consumed a considerable amount of time. I faced issues with inconsistent labeling and had to develop scripts to rectify these discrepancies.

Training the model presented its own set of challenges. I grappled with overfitting, where the model performed exceptionally well on the training data but poorly on validation data. This required me to experiment with various regularization techniques and data augmentation strategies to improve generalization.

2. How AI Tools Were Used for Coding Assistance
Throughout the project, I leveraged several AI tools to assist me during development:

Jules (Google AI Assistant): I frequently used Jules to get quick clarifications on PyTorch concepts, architecture design, and debugging tips. It was particularly helpful when I hit blockers and needed concise, contextual support without switching tools.

ChatGPT: Served as an external reference when I wanted deeper explanations or alternative ways to solve a coding problem. It was especially useful in understanding loss functions and handling bounding boxes.

GitHub Copilot: Integrated with my editor, Copilot helped write repetitive code faster, like defining dataset classes or utility functions. However, I made sure to always review its suggestions, as they weren’t always accurate.

Overall, these AI tools significantly accelerated my workflow. The combination of quick insights from Jules and broader explanations from ChatGPT made learning more fluid and contextual.

3. What I Learned From the Project
This assignment offered hands-on exposure to building an end-to-end object detection model. Some of the key things I learned include:

Deep Learning Concepts: Especially around convolutional neural networks and SSD architecture.

PyTorch Workflows: I got much more comfortable with writing custom datasets, managing training loops, and debugging model behavior.

Data Challenges: I learned the importance of properly labeled data and how small issues in annotations can break training.

Evaluation Techniques: I gained familiarity with metrics like mAP, IoU, and loss curves, and how they reflect model performance.

More than anything, I learned how to troubleshoot efficiently using both documentation and AI assistance — a real-world skill I’ll carry forward.

4. What Surprised Me About the Process
I was surprised at how challenging it was to get a model to generalize well on such a small dataset. I initially thought a model would train decently with limited data, but this project reinforced the importance of dataset size and variability.

Also, I was impressed by how much smarter AI tools like Jules and ChatGPT have become. Sometimes they provided better debugging help than searching through Stack Overflow or GitHub issues. That was unexpected and incredibly helpful.

5. Balance Between Writing Code Myself vs. Using AI Assistance
I feel that I struck a good balance. I relied on AI tools like Jules and ChatGPT to help me understand or scaffold tricky parts of the code, but I always rewrote and customized the final implementation myself.

Rather than using AI as a shortcut, I used it as a thought partner — to explain concepts, suggest improvements, and catch errors I might have missed. This collaborative approach helped me learn faster while still staying hands-on with the project.

6. Suggestions for Improving This Assignment
Here are a few ideas that could improve the assignment for future interns:

More Clarification on Model Expectations: For example, whether we’re supposed to hit a specific accuracy or just build a functioning pipeline.

AI Usage Guidelines: Since many of us are using tools like Jules or ChatGPT, it might help to include guidance on how to use AI responsibly during the project.

Optional Walkthrough Video: A short video explaining the expected pipeline (dataset → model → training → evaluation) could reduce the early confusion and save time.

Debugging Tips Section: A section in the README on common errors (e.g., shape mismatches, data loading issues) would be really helpful.
