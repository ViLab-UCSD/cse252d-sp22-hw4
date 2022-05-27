# CSE252D Spring 2022 HW4

### Instructions

1. Attempt all questions.
2. Please comment your code adequately.
3. Include all relevant information such as text answers, output images in the Jupyter notebook.
4. **Academic integrity:** The homework must be completed individually.
5. Please select pages for each answer on Gradescope to facilitate smooth grading.
6. **Due date:** **Wed Jun 8, 11:59PM PDT**.
7. Access the HW by cloning this repository using,\
     ``git clone https://github.com/ViLab-UCSD/cse252d-sp22-hw4.git``
8. Follow the Jupyter Notebook ``hw4_questions.ipynb``.
9. Follow the rest of README (this file) for instructions on how to setup your environment, data and compute.
10. Submit the PDF version of your notebook and your code on **Gradescope**.\
    (1) Convert the ipynb file to **pdf** and upload it to **Homework 4 writeup**. Select pages for each answer. Only include images, code, and text required by the questions in this file. \
    (2) Compress any supporting documents you think is necessary to justify your reasoning and answers into a **zip** file and
    upload it to **Homework 4 code**. Do not include any dataset, model or large data files.
11.  Rename your submission files as `Lastname_Firstname.pdf` and `Lastname_Firstname.zip`.

### Frequently Asked Questions (FAQ)
**I don't have access to a GPU. What are my options?**\
You can use the Data Science & Machine Learning Platform (DSMLP). It is a
Kubernetes cluster with well defined usage policies and an easy-to-use
interface. You can request for ephemeral pods/containers with a persistent
storage space (your home directory). There are two ways to do this:
1. Use the GUI at [Datahub](https://datahub.ucsd.edu).
2. Use a deployment script. Follow [these](https://github.com/ViLab-UCSD/cse252d-sp22-hw1#22-option-2-recommended-on-data-science--machine-learning-platform) intructions from HW1.

**Where can I get the data?**\
The `/datasets/cs252d-sp22-a00-public/hw4_data/` directory on DSMLP contains the
required data for this HW. If you are using the cluster, change the paths in the
Jupyter notebook such that they point to the correct directories. If you are
using your own machine, you can use `scp` or `rsync` to download the data.
You can also access it from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit=).

**How do I monitor my training progress on the cluster?**\
You can use tensorboard. Follow the instructions on [this](https://piazza.com/class/l122522417k46?cid=43) Piazza post on how to do
this. Alternatively, we provide an easy to use bash script:
1. Remote: `/datasets/cs252d-sp22-a00-public/tb.sh {logdir} {port-number}`.
2. Local: `ssh -N -L localhost:{port-number}:127.0.0.1:{port-number} {user}@dsmlp-login.ucsd.edu`.
3. Navigate to `localhost:{port-number}` in your browser.

**How do I run my pods in background?**\
You can use the `-b` flag when you use the launch script. If you want to launch
Jupyter sessions in background, use
`/datasets/cs252d-sp22-a00-public/cs252d-scipy-ml.sh` instead of
`launch-scipy-ml.sh`. Alternatively, use tmux.

`module` **not found. How can I install the dependencies?**\
Use `requirements.txt` to install the dependencies\
`pip install -r requirements.txt`.

---
More to come. We will update this section as and when we receive questions
on Piazza.
