from setuptools import setup, find_packages

setup(
        name='minjaerl',
        version="0.0.1",
        description=(
            'minjae-rl'
        ),
        author='Minjae Cho',
        author_email='minjae5@illinois.edu',
        maintainer='Mgineer117',
        packages=find_packages(),
        platforms=["all"],
        install_requires=[
            "gym>=0.15.4,<=0.26.2",
            "mujoco==2.3.3",
            "cython<3",
            "cvxpy",
            "matplotlib",
            "h5py",
            "opencv-python",
            "numpy<1.24",
            "pandas",
            "scikit-learn",
            #"ray==1.13.0",
            "torch==2.3.0",
            "seaborn",
            "oauthlib>=3.0.0",
            "tensorboard",
            "tqdm",
            "wandb"
        ]
    )
