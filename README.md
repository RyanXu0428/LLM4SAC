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


### Dependencies Installation

Before running the code, ensure that you install the necessary dependencies:

1. **VRX**: A robotics simulation framework that allows testing and development in virtual environments.
   - Clone the repository and follow the installation instructions:
     ```bash
     git clone https://github.com/osrf/vrx.git
     ```

2. **OpenAI ROS**: The OpenAI Robotics library for integrating with ROS (Robot Operating System).
   - Clone the repository and follow the installation instructions:
     ```bash
     git clone https://github.com/edowson/openai_ros.git
     ```

## Running the Code

Once the required dependencies are installed and the models are set up, you can execute the program by running the following command:

```bash
python main.py
```

