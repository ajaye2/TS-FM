# TS-FM

### Install conda
Follow these steps to install Miniconda on Ubuntu:

Open the terminal.
Download the Miniconda installer script by running the following command:
For Python 3.x:

arduino
Copy code
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
For Python 2.x (less recommended, as Python 2.x is no longer officially supported):

arduino
Copy code
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
Make the installer script executable:
For Python 3.x:

bash
Copy code
chmod +x Miniconda3-latest-Linux-x86_64.sh
For Python 2.x:

bash
Copy code
chmod +x Miniconda2-latest-Linux-x86_64.sh
Run the installer script:
For Python 3.x:

Copy code
./Miniconda3-latest-Linux-x86_64.sh
For Python 2.x:

Copy code
./Miniconda2-latest-Linux-x86_64.sh
Follow the prompts during the installation process. It is recommended to accept the default settings, which will install Miniconda in your home directory.
After the installation is complete, close and reopen the terminal. You should now have Conda installed on your system.
To verify that Conda is installed and working correctly, run:

css
Copy code
conda --version
This command should display the installed Conda version.

Now you can create and manage Conda environments, as well as install packages using the conda command.