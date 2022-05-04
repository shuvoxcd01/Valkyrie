# Valkyrie

### About
A population based approach to train reinforcement learning agents to play Atari 2600 games. 
The targeted environment here is Pong-v4. **The agent achieves state of the art result on this environment.**

<img src="https://user-images.githubusercontent.com/16299215/166633629-0cb29ab6-0bb2-4e72-97fc-c6f57c94b78c.png" width="300" />

---

### Python Version
Python 3.8

---

### Environment Setup & Experiment Steps
1. Clone the repository.  
`git clone https://github.com/shuvoxcd01/Valkyrie.git`  
2. Change directory to Valkyrie/  
`cd Valkyrie`  
3. Now, try one of the following methods.  

- **Method 1**  
The easiest way to create the experimental environment is to use a docker container as the dev environment.
1. Pull image from docker hub.  
`docker pull shuvoxcd01/valkyrie` if you have GPU support in your machine. [*You would also need [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) for the GPU to work.*]   
`docker pull shuvoxcd01/valkyrie-no-gpu` if you don't want GPU support.  
2. Run container with the local 'Valkyrie' directory mounted.  
`docker run -it --gpus all -v $PWD:/home/Valkyrie shuvoxcd01/valkyrie bash` [With GPU support]  
`docker run -it -v $PWD:/home/Valkyrie shuvoxcd01/valkyrie-no-gpu bash`  [Without GPU]
3. Train an agent from terminal. The prompt should be in `/home` directory within Valkyrie (container) bash.   
`cd Valkyrie`  
`python src/train_pong.py`  


- **Method 2**  
*This method would fail if any system library on which the runtime depends is missing. This method is **NOT Recommended**.*
1. Create a virtual environment and install dependencies from `requirements.txt`.  
`python3.8 -m venv venv`  
`pip install -r requirements.txt`
2. Run `train_pong.py`.  
`python src/train_pong.py`

### Still facing problem?
You can build docker image from the dockerfile(s) provided inside the  `dockerfile-dev-env` directory.    
Use `dockerfile-dev-env/with-gpu/dockerfile` to build image with GPU support.   
Use `dockerfile-dev-env/without-gpu/dockerfile` to build image without GPU support.    
Change directory into either `dockerfile-dev-env/with-gpu/` or `dockerfile-dev-env/without-gpu/` and then run the following command.    
`docker build .`    
Then use this image to run a container and use that as the dev environment.

