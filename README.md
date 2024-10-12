# LLM4SAC
---
Autonomous docking task in a real-world scenario. The USV begins from the pre-docking zone and autonomously maneuvers towards the docker


<img src="./gif/usv_docking_mission.png" width="1000"/>


---
## Real-World Experiment Results 

We conducted 20 real-world experiments using our algorithm, achieving 16 successes and 4 failures. Below, you can see a visual demonstration of both the successful and failed experiments.

<table> <tr> <!-- 第一行，成功实验的 GIF --> <td align="center"> <img src="./gif/success.gif" width="800"/><br> <strong>Success Video</strong> </td> </tr> <tr> <!-- 第二行，失败实验的 GIF --> <td align="center"> <img src="./gif/fail.gif" width="800"/><br> <strong>Failed Video</strong> </td> </tr> </table>


## Installation Requirements

This project was developed and tested on the following system:

- **Operating System**: Ubuntu 20.04
- **ROS Version**: Noetic


Additionally, this project utilizes locally deployed models using **Ollama**:

- **Models Used**:
  - LLaMA 3B
  - Gemini 2 7B
 
  - 
Before running the code, make sure to install the following dependencies:

1. **VRX**: A robotics simulation framework that allows testing and development in virtual environments.
2. **OpenAI ROS**: The OpenAI Robotics library for integrating with ROS (Robot Operating System).

You can install these packages using the following commands:

```bash
# Install VRX
git clone https://github.com/osrf/vrx.git
# Follow the installation instructions in the VRX repository

# Install OpenAI ROS
git clone https://github.com/edowson/openai_ros.git
# Follow the installation instructions in the OpenAI ROS repository

##Running the Code
After installing the required dependencies and ensuring that the models are properly set up, you can run the main script to execute the program:
```bash
python main.py


