from utils import registerEnvs
import gym
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    custom_xml = 'environments/hoppers'
    registerEnvs(None, 1_000, custom_xml)
    env_names = [f'hopper_{i+3}' for i in range(0, 3)]
    envs = []
    for env_name in env_names:
        e = gym.make("environments:%s-v0" % env_name)
        e.reset()
        for i in range(10):
            e.step(e.action_space.sample())

        envs.append(e)

    imgs = []
    for env in envs:
        img = env.render(mode='rgb_array')
        imgs.append(img)

    # Specify the path to your home directory
    home_directory = os.path.expanduser("~")

    # Assuming 'imgs' is a list of image arrays
    for idx, img in enumerate(imgs):
        plt.imshow(img)
        # Define the filename for each image
        filename = os.path.join(home_directory, f"image_{idx}.png")
        # Save the image to the specified path
        plt.savefig(filename)
        # Clear the current figure to prepare for the next image
        plt.clf()

    print("DONE")