<h1 align='center'> Real Time Face Mask Detection </h1>


# <a name="_pikyha6s2ic"></a>**Chapter 1**

# <a name="_l7dkklt0sa7o"></a>**PROBLEM STATEMENT**

In a pandemic where there is a virus that spreads very rapidly, wearing a mask has become very crucial. Lots of people are serious about this but sometimes people forget to wear the mask. Hence there is a need to identify whether a person is wearing a mask or not, so that appropriate actions can be taken.

The main idea behind this project is to develop a module that is capable of detecting whether a person is wearing a mask or not. This module can be built upon and can be used by various organisations like hospitals, schools, universities, etc. It can also be integrated with other modules for achieving the desired results. One application of this module can be in a situation where several people are sitting and if any person removes the mask, a message can be shown on the screen or a beep sound can be made to indicate that there is a violation of the rule. Also, a rectangle can be drawn around the face of the person on the screen to identify the defaulter.

Once this module is written, it can be easily reused by any organisation that is looking for a solution like this. It can easily be used by anyone, they don’t need to learn, write the code, train the model, or hire someone else. This module is built upon python which is very popular and easy to learn, people can easily modify and build upon this application to solve their specific problems, they won’t have to write code from scratch.

There are similar projects available, some have implemented it on Raspberry Pi [1], while some have developed a single-stage tool and tested it for different datasets [2]. This project is focusing only on building a module that recognises whether the person is wearing a mask properly or not. Later on, it can be forked to provide more advanced facilities.

This module is written in python which is easy to understand. Code will be documented. Even the source code will be available on the internet so that people can add things to this module and tweak the program according to their needs.

This project will use two popular algorithms and libraries:

1. CNN (Convolutional Neural Network) - It is a neural network algorithm that is used to extract higher representations for the image content.
1. TensorFlow - It is a Machine Learning framework mainly used for training models.

# <a name="_31w0fu98j3sj"></a>**Chapter 2**

# <a name="_mybcppzayebc"></a>**PROCESS MODEL**

A Process Model provides a roadmap and defines the flow of all activities, actions, and tasks, the degree of iteration, the work products, and the organization of the work that must be done in Software Engineering Work [3].

This Project will use the **Incremental Process Model** as shown in *Fig 2.1*. In this model, both linear and parallel process flows are combined. In each linear sequence, an increment of the Software is delivered.

Here, the first increment is the core product. In later increments, the supplementary features are deployed. After each increment, the product is tested and evaluated. Further, the plan for the next increment is made. The whole process repeats again and again for each increment until the final product is completed [3].

This model is beneficial when the initial software requirements are known. With the help of these requirements, the core software is completed and deployed. Without having complete software requirements, limited software can be made available to use in a short period of time. Later on, increments can be made to add more features to the software.

**Fig. 2.1.** The Incremental Process Model

Even for this software, the first increment will be the core product i.e. the base Machine Learning Model for Face Mask Recognition will be deployed. In further increments, the model will be trained and tested using the dataset. In the last increment, the model will be deployed after testing it with real-world data so that data leakage can be prevented.

The project can be later on expanded and implemented to use in various places. Once the base software module for Face Mask Detection is completed and deployed, it can be further incremented to embed with the CCTV Camera of the College. It will be achieved due to the Incremental Process Model.

# <a name="_czncgdzfgidu"></a>**Chapter 3**

# <a name="_2yi69g9a6avb"></a>**Requirement Analysis & Modeling**

The goal of requirement engineering is to produce a document called Software Requirements Specification (SRS). This document is written in a way that is easy to understand. It describes what the system will do rather than how it will do it.
# <a name="_7tgq7dumlbsu"></a>**3.1.	DFD**
DFD or Data Flow Diagrams are used for modelling the requirements. DFD shows the flow of data within the system.

In DFD various shapes can be used to make the interpretation of the diagram easier. The arrow represents the data flow. The oval shape represents the process. The rectangle represents the source or the sink of data and parallel lines represent a data repository [4].
## <a name="_cu2c5f2f76hy"></a>**3.1.1.	Context Level DFD**
The Context Level DFD also called Level 0 DFD is a simple model that gives a basic overview of the whole system or process being modelled. It gives the relationship of the system with the external entities [4] as shown in *Fig 3.1.1*.

**Fig 3.1.1.** Context Level DFD
## <a name="_bikbn8bqdom2"></a>**3.1.2.	Level 1 DFD**
Level 1 DFD is an elaborated version of Level 0 DFD. It also describes the subprocesses and how they are relating to the external entities and to each other [4] as shown in *Fig 3.1.2*.

DFD for this project consists of two parts. The first part deals with the training of the model for Face Mask Detection. First, the image dataset is loaded and organized in order to train the model. Once the model is trained, it is stored on the disk to use in the module.

The second part of DFD deals with detecting whether the person is wearing the mask or not. It loads the model from the disk and takes input from the camera. The image from the camera is processed and passed to the model for prediction. Based upon this prediction suitable output is shown on screen or suitable sound is produced.

**Fig 3.1.2.** Level 1 DFD
## <a name="_qzhgwg8ovonb"></a>**3.1.3.	Level 2 DFD**
Level 2 DFD is an elaborated version of Level 1 DFD. It also describes the subprocesses and how they are relating to the external entities and to each other in more detail.

As the DFD for this model is divided into two parts, the first one is for training while the second one is for detection as shown in *Fig 3.1.3*.

In the first part, the image dataset is loaded and organized. Further, the images are converted into mathematical form i.e NumPy arrays. CNN model is used for better image feature extraction. Another list is maintained for corresponding images. Now, the model is trained. This model is trained using TensorFlow. Once this model is trained, the accuracy of the model is tested. If the result is satisfactory, the model can be approved and saved to the disk for future use.

**Fig 3.1.3.** Level 2 DFD
# <a name="_wxh704asotbf"></a>**3.2.	Data Dictionary**
Data dictionaries are repositories of information about all data items defined in DFDs. The data dictionary holds records about other objects in the database, such as the name of data items, aliases, descriptions, related data items, range of values, etc as described in *Table 3.2*.

**Table 3.2.** Data Dictionary

|**Field Name**|**Data Type**|**Description**|
| :- | :- | :- |
|Raw Images|.png or .jpg|Image Captured from Dataset|
|Image in mathematical form|NumPy array|Converted form of Image into array form|
|model|.model file|Trained model for mask detection|
|‘data’|NumPy array|Array storing data of all images|
|‘label’|Array of String|(previous image)|
|record/show accuracy|tuple|Holds the model accuracy information|
|Input to model|Numpy array|Image (taken from camera source) converted into NumPy array|
|username|First name + Last name + 3 digit code|Where First name, Last name are varchar sequences. It is combined with 3 digit code to make it unique.|
|password|Characters + Digits + Special characters|Used to authenticate users. Combination of characters, digits, and special characters|
# <a name="_3lkyc5k3iiiw"></a>**3.3.	Use Case Diagram**
A use case diagram visually represents what happens when an actor interacts with the system. It, therefore, represents what the system will do based on the type of interaction [4].

There are three components of a use case diagram namely actor, use case, and relationship between actors and use case and/or between the use cases. Actors appear outside of the rectangle since they are external to the system. Use cases appear within the rectangle, providing functionality. This rectangle represents a system. A relationship is represented by a solid line between an actor and each use case in which the actor participates.

The use case diagram for this project is shown in *Fig 3.3*. There are two actors: the user and the admin. As shown in the figure, only the admin has the privilege to enable the mask detection system. Also, the admin has privileges to access or control all the functionalities like image processing, model training, and output configuration. The user can show his/her face in the camera, this will act as an input to the system and then the system will do the necessary processing and generate output accordingly.

**Fig 3.3.** Use Case Diagram
# <a name="_872qi2pduvt6"></a>**3.4.	Sequence Diagram**
Sequence Diagram shows the object interactions arranged in a time sequence. It depicts the objects involved in the scenario and the sequence of messages exchanged between the objects needed to carry out the functionality of the scenario. Its main aim is to show object interactions arranged in time sequence. A proper Sequence Diagram contains the following components [5]:

- Actors (User)
- Message (methods)
- Return value (in response to the previous message)
- Indication of loops or iteration area

This work contains two sequence diagrams. The first one is for the login module while the other one is for the face mask detection system.

*Fig 3.4(a)* represents the sequence diagram of the login module and *Fig 3.4(b)* depicts the steps that are performed in their order while the software is running.

**Fig. 3.4(a).** Sequence Diagram - Login [4]

**Fig. 3.4(b).** Sequence Diagram - Face Mask Detection

# <a name="_a6qard9v8ua"></a>**Chapter 4**

# <a name="_tc69yxl766o6"></a>**Software Requirement Specification**

SRS or Software Requirement Specification specifies what the software will do. It is a formal document and acts as a contract between the client and developers.
# <a name="_yt2ifro06uri"></a>**4.1.	Overall Description**
The main objective of this project is to train and deploy a Face Mask Recognition model so that it can be used to identify whether people are wearing a mask or not.
## <a name="_5khu8vvg9ugf"></a>**4.1.1. Product Functions**
The main function of the model is to take input from the camera and detect whether the person is wearing a mask or not and generate a suitable response.
## <a name="_4xi1mmzd4uek"></a>**4.1.2. User Characteristics**
Any person who knows how to operate a computer would easily be able to run the program. Even if the user is not familiar with programming and computing can be easily trained to use this program.
## <a name="_qwnlxsf0fzpb"></a>**4.1.3. General Constraints**
This model requires a webcam or external camera to take pictures from the real world and detect masks. It also requires good enough hardware to run upon. Exact hardware constraints are mentioned in section 4.5.2
## <a name="_t14urdmuvxjs"></a>**4.1.4. Assumptions and Dependencies**
The programming is done in Python using TensorFlow modules. The major dependencies are listed below.

- TensorFlow
- OpenCV

This program can run on Windows, macOS, and Linux provided that they have installed the above-mentioned libraries. A full list of requirements is also provided which can be easily installed.

This project depends upon various libraries for eg., TensorFlow, as a natural consequence of the program might break when dependent libraries update. However this is inevitable, issues like this are easier to fix and the chances of happening are very less.
# <a name="_4of69a9lm2lf"></a>**4.2.	External Interface Requirements**
Sometimes a system needs to interact with the outside world in that case some interfaces are needed and hence these requirements are called external interface requirements.
## <a name="_5dbnywltv6sj"></a>**4.2.1. Product Functions**
To use the Mask Detection program the user must authenticate himself/herself then only he/she can access the program.
## <a name="_zb6kx57qjbec"></a>**4.2.2. Hardware Interface**
This requires a camera (webcam or external) to be able to detect the mask.
## <a name="_xl4besrnnjj9"></a>**4.2.3. Software Interface**
This program uses external libraries which include TensorFlow, Keras, OpenCV, etc.
# <a name="_itzl0y3o9wzk"></a>**4.3.	Functional Requirements**
Functional requirements focus on functions that are performed by the system or software. These are also called product features.
## <a name="_g0nv29fgzxfq"></a>**4.3.1. Login Requirement**
The command-line login interface should be provided so that the admin can log in to the system and start this program.
## <a name="_3skua9p2rjzb"></a>**4.3.2. Mask Recognition Requirement**
Once the user gets successfully logged in, he/she must be able to run the program. The interface or steps of running the program should be easy and simple to understand.
# <a name="_hh1hs7c3mjj3"></a>**4.4.	Performance Requirements**
Performance requirements specify a set of criteria that defines how things should perform and the conditions necessary for the proper functioning of the software.
## <a name="_67c9v13q8wxl"></a>**4.4.1. Mask Detection Probability**
If the hardware and software that are required are available then the software should be able to detect the masks that are in the current frame of the camera. It should be able to detect at least 3 to 5 persons in the camera (if there are enough persons in the camera) with an accuracy of at least 80 per cent.
# <a name="_6djko8rgah8o"></a>**4.5.	Design constraints**
Every software has some constraints, possible constraints for this project are listed below.
## <a name="_89z8zol465bu"></a>**4.5.1 Dataset**
The accuracy of the model also depends upon the quality of the dataset on which it is trained.
## <a name="_tnil5ue8eula"></a>**4.5.2 Hardware Constraints**
Software of this kind is heavy on the system because there is a lot of work going on the backend.

This model also requires a computer with at least 4 GB of RAM and an i3 processor or equivalent for the smooth running of the program.

Training a model requires a computer with a graphic card or online services like Google Colab can also be used.

This software requires a camera in order to take input hence a good quality camera is essential for the proper working of the software.

This software also requires a screen on which the output will be shown.

# <a name="_laycia6i19jb"></a>**Chapter 5**

# <a name="_edccvb8uqtze"></a>**Estimations**

In order to measure the software or product quantitatively it is essential to measure the overall size of the software so that overall efforts and cost can be calculated. There are various techniques available and one of them is Function Points.
# <a name="_sq66y15zf3q6"></a>**5.1.	Function Points**
Function Point (FP) can be used to measure the functionality or complexity of the system from that effort and cost can be derived.

Function points are based on five Information Domain Values which are listed in *Table 5.1*.

**Table 5.1.** Information Domain Values

|Information Domain Values|Count|Weighting Factor|Total|
| :- | :- | :- | :- |
|External Inputs (EI’s)|1|6|6|
|External Outputs (EO’s)|1|7|7|
|External Inquiries (EQ’s)|0|6|0|
|Internal Logical Files (ILF’s)|1|15|15|
|External Interface Files (EIF’s)|2|10|20|
|Count Total|48|

To compute Function Points (FP), the following formula is used:

FP = count total \* [0.65 + 0.01 \* 𝝨(F<sub>i</sub>)]

Where the count total is the sum of all FP entries obtained from *Table 5.1*. 𝝨(F<sub>i</sub>) is the sum of all responses to the question in *Table 5.2*. It is called Value Adjustment Factors (VAF).

**Table 5.2.** Value Adjustment Factors

|S.No.|Question|Response|
| :- | :- | :- |
|1|Does the system require reliable backup and recovery?|0|
|2|Are specialized data communications required to transfer information to or from the application?|0|
|3|Are there distributed processing functions?|0|
|4|Is performance critical?|4|
|5|<p>Will the system run in an existing, heavily utilized operational environment?</p><p></p>|2|
|6|Does the system require online data entry?|0|
|7|Does the online data entry require the input transaction to be built over multiple screens or operations?|0|
|8|Are the ILFs updated online?|0|
|9|Are the inputs, outputs, files, or inquiries complex?|4|
|10|Is the internal processing complex?|5|
|11|Is the code designed to be reusable?|4|
|12|Are conversion and installation included in the design?|0|
|13|Is the system designed for multiple installations in different organizations?|5|
|14|Is the application designed to facilitate change and ease of use by the user?|4|

Based on the above results following calculations are done:

Count Total = 48
𝝨(F<sub>i</sub>) = 28
FP = Count Total \* [0.65 + 0.01 \* 𝝨(F<sub>i</sub>)]
FP = 48 \* [0.65 + 0.01 \* 28]

|FP = 44.64|
| :- |
# <a name="_nfbke6xq4z1"></a>**5.2.	Effort Estimation**
Effort describes the estimated amount of effort required to develop the software. *Table 5.3* specifies productivity on the basis of developers' experience and capability.

**Table 5.3.** Productivity Chart

|Developer’s experience/capability|Very Low|Low|Nominal|High|Very High|
| :- | :- | :- | :- | :- | :- |
|Environment maturity/capability|Very Low|Low|Nominal|High|Very High|
|PROD|4|7|13|25|50|

PROD = 7 FP/pm
Effort =FPPROD
Effort =44.647

|Effort = 6.38 pm|
| :- |
# <a name="_dsbbm02fi3cz"></a>**5.3.	Cost**
Cost refers to the approximated cost of the project. It is calculated based on the efforts that the project requires.

Cost = Effort (pm) \* Labour Rate (INR/pm)
Cost = 6.38 \* 1,00,000

|Cost = 6,38,000 INR|
| :- |



# <a name="_4s61zy8b328v"></a>**Chapter 6**

# <a name="_qdfv404nldx3"></a>**Scheduling**

Scheduling is an important part of software planning. It gives an overview of what needs to be done and in how much time so that the project could be completed in time.
# <a name="_hmtw2i3j99g4"></a>**6.1.	Timeline Chart**
Timeline chart is a visual representation of a series of events. It describes what event or action will take place and for how much duration. *Fig 6.1 (a)* and *Fig 6.1 (b)* show the Timeline chart for this work.

**Fig 6.1(a).** Timeline Chart till Week 5

**Fig 6.1(b).** Timeline Chart from Week 6

# <a name="_tewedn42fqgy"></a>**Chapter 7**

# <a name="_j10xt3flvnpu"></a>**Risk Management**

Risk management is the process of identifying, assessing, and controlling threats to a project or stakeholder resources. A proactive risk management strategy is always helpful. Risk management involves identifying the risk and accessing its probability of occurrence and its impact if that risk becomes the reality.
# <a name="_hbfa7uncpyc8"></a>**7.1.	Risk Table**
Risk Table is a table that gives simple techniques for risk projection. *Table 7.1. (a)* describes the risks associated with this work and *Table 7.1. (b)* gives the key to abbreviations.

**Table 7.1. (a)** Risk Table

|**S.N.**|**Risks**|**Category**|**Probability**|**Impact**|**RMMM**|
| :- | :- | :- | :- | :- | :- |
|1|Team members may be inexperienced|ST|80%|Critical|The team members will need more time to learn the technologies used for the project.|
|2|Dependencies may not be available on the end-user system|TE|75%|Catastrophic|A ***requirement.txt*** file will be provided along with the module. Running the command ***pip -r requirement.txt*** will install all dependencies.|
|3|The outside library may break/ function path in the library may get changed|TE|70%|Catastrophic|To fix it, the path of the function in the library will have to change in the import statement in the code. It will depend on the changes made to the libraries used for this project.|
|4|Users may face difficulty in using the module|TE|50%|Critical|A well-documented ***README.MD*** file will be provided along with the module.|
|5|The model may take more hardware resources than expected|TE|30%|Marginal|A better system will be needed to use the module at that time. The module will be improved later to make it consume fewer resources.|
|6|Camera may fail to capture images|TE|25%|Catastrophic|Need to change the camera at short notice.|
|7|A team member may leave the project|ST|20%|Critical|All team members will have to know about every module of the project in brief so that the work can be continued in such cases.|
|8|Performance may not be up to the mark|TE|25%|Marginal|The model will be trained with a diverse set of image datasets to improve the performance.|
|9|Cost Estimates may be low|CU|20%|Marginal|Price-effective options should be preferred from the beginning.|
|10|Project may not complete on time|BU|10%|Negligible|Adhere to complete the project on time.|

**Table 7.1. (b)** Key to Abbreviation

|**Abbreviation**|**Keyword**|
| :-: | :-: |
|ST|Staff Risk|
|TE|Technical Risk|
|CU|Cost Risk|
|BU|Business Risk|



# <a name="_nrayjawel529"></a>**Chapter 8**

# <a name="_up3lfu5y1urs"></a>**Design**

Design is the implementation of features specified in SRS (Software Requirements Specifications). Design helps in developing software that is functional, reliable, easy to develop, understand, test, modify, and maintain.
# <a name="_m7jcqfnv4uw6"></a>**8.1.	Structural Chart**
Structural Chart is a chart that is used to show the breakdown of a system. It is mainly used to arrange the program modules of the system into a tree. Each module is shown using a box and they are connected to visualize the relationship between all modules of the system. *Fig 8.1.* gives the Structural Chart for this work.

**Fig. 8.1.** Structural Chart
# <a name="_bwgznybzcfrn"></a>**8.2.	Pseudo Code**
Pseudo Code is simple and plain sentences that describe the algorithm or the code of the program.

def main():

`    `load face detection model

`    `load mask detection model

`    `load camera

`    `while(True):

`        `input()

`        `detectFace(frame)

`        `predictMask((faces, locations))

`        `labelFace((locations, predictions))

EndWhile

def input():

`    `Read camera frames with cv

`    `Resize frames to 400 x 400

`    `return frame

def detectFace():

`    `faceList = faceModel.detectFaces()

`    `for face in facelist:

`        `if (confidence > 0.5)

`            `get the bounding box for each face

`            `ensure bounding box is within frame)

`                `resize the frame

`                `convert from BGR to RGB

`                `preprocess the image

`                `append it to list of faces

`                `append corresponding location to list of locations

`        `EndIf

`    `EndFor

`    `retun (faces, locations)

def predictMask((faces, locations)):

`    `predictions = MaskNet.predict(faces)

`    `return (locations, predictions)

def labelFace((locations, predictions)):

`    `for each face and prediction:

`        `if (maskProb > noMaskProb):

`            `label = "With Mask"

`            `color = Green

`        `else:

`            `label = "Without Mask"

`            `color = Red

`        `EndIf

`        `Draw rectangle around face

`        `Draw label above rectangle

`    `EndFor

`    `Show the labelled frame

# <a name="_yfxep2uq80dc"></a>**Chapter 9**
# **Coding**
You can check out the codes in the repository.

Fig. 9 (a) represents the Training Accuracy/Loss graph while Fig. 9 (b) shows the Testing Accuracy/Loss graph. The expected output of this work is shown in Fig 9 (c) and Fig 9 (d).

Fig. 9. (a) Training Accuracy/Loss Graph

Fig. 9. (b) Testing Accuracy/Loss Graph

**Fig. 9 (c).**  With Mask Output

**Fig. 9 (c).**  Without Mask Output



# <a name="_2zcu3dj3xvwo"></a>**Chapter 10**

# <a name="_j0tdj4sensjs"></a>**Testing**

Testing is the process of verifying the developed software against the Software Requirement Specification (SRS). It is done with the intent of finding errors or bugs in the software so that the end-user gets good quality software.

It comprises verification and validation. Verification is applied to the early stages of the software development life cycle like planning, software requirement specification, etc. Validation is generally applied in later stages of development where the software is executed and tested in order to ensure that the product built actually performs the way it is expected to perform.

**Black Box Testing:** In black-box testing, the code of the program is not checked, rather a set of inputs is passed and if the output matches the expected output then the test case is passed.

**White Box Testing:** In white-box testing, the internal structure of the program is also considered and test cases are designed accordingly.
# <a name="_tk21y91j6zw2"></a>**10.1.	Test Case Design**
A test case is designed with the intent of finding errors or bugs in the software. A test case specifies the inputs, the expected output, and the actual output. If the output given by the software matches the output expected then the software passes that test case.
## <a name="_kb9ocae8bhrm"></a>**10.1.1. Boundary Value Analysis**
Faults generally occur at the boundary values hence it is essential to test the software at nominal as well as boundary values. In this project, there are no simple mathematical values like 5 or 10. Hence the robustness of the software is being tested by testing the software at edge cases.

X: Person wearing a face mask

Y: Person not wearing a face mask

Z: Not a human face

Variables X and Y assume that a human is standing in front of the camera. Their value can range from (facing 20 degrees left to 20 degrees right and 20 degrees top to 20 degrees bottom).

All the test cases were performed and their output is recorded.

*Tables 10.1.1., 10.1.2.,* and *10.1.3.* shows test cases with face masks, without face masks, and with no person standing in front of the camera respectively.

**Table 10.1.1.** Test cases while the person is wearing a face mask

|**S.No**|**Test Case**|**Expected Output**|**Actual Output**|
| :- | :- | :- | :- |
|1\.|20 degrees left|With mask|With mask|
|2\.|10 degrees left|With mask|With mask|
|3\.|In center|With mask|With mask|
|4\.|10 degrees right|With mask|With mask|
|5\.|20 degrees right|With mask|With mask|
|6\.|20 degrees top|With mask|With mask|
|7\.|10 degree top|With mask|With mask|
|8\.|10 degrees bottom|With mask|With mask|
|9\.|20 degrees bottom|With mask|With mask|

**Table 10.1.2.** Test cases while the person is not wearing a face mask

|**S.No**|**Test Case**|**Expected Output**|**Actual Output**|
| :- | :- | :- | :- |
|1\.|20 degrees left|Without mask|Without mask|
|2\.|10 degrees left|Without mask|Without mask|
|3\.|In center|Without mask|Without mask|
|4\.|10 degrees right|Without mask|Without mask|
|5\.|20 degrees right|Without mask|Without mask|
|6\.|20 degrees top|Without mask|Without mask|
|7\.|10 degree top|Without mask|Without mask|
|8\.|10 degrees bottom|Without mask|Without mask|
|9\.|20 degrees bottom|Without mask|Without mask|

**Table 10.1.3.** Test case when there is no person standing in front of camera

|**S.No**|**Test Case**|**Expected Output**|**Actual Output**|
| :- | :- | :- | :- |
|1\.|No person standing|No labeling|No labeling|
# <a name="_nzcfdyriwdbu"></a>**10.2.	Flow Graph**
Flow graph is a visual representation of control flow in the code. They represent the flow inside the program. Flow graphs for various modules in the project are given below in *Fig. 10.2 (a)* and *Fig. 10.2 (b).*

**Fig10.2.(a)** Flow Graph for detect\_face() method


**Fig10.2.(b)** Flow Graph for label\_face() method￼
# <a name="_cw337kuk41j1"></a>**10.3	Basis Path Set**
White box testing is done by considering the internal structure of the program and various paths the program can follow. Below is the list of various individual paths that can be followed by each function.
## <a name="_ypsqnwzi7310"></a>**10.3.1 detect\_face()**
1. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 3, 15
1. 1, 2, 3, 4, 13, 14, 3, 15
1. 1, 2, 3, 15
## <a name="_99iqw7tq6mgj"></a>**10.3.2. label\_face()**
1. 1, 2, 3, 4, 5, 9, 10, 11, 12, 2, 13
1. 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 2, 13
1. 1, 2, 13
# <a name="_c1htasfhl9uc"></a>**10.4.	Cyclomatic Complexity**
Cyclomatic complexity is a software metric used to indicate the complexity of the program. It is a quantitative measure of the number of linearly independent paths the program can follow in a source code.

Cyclomatic complexity can also be calculated by a mathematical formula.

V(G) = E - N + 2, where E is the number of edges and N is equal to the number of nodes
## <a name="_4afwas5dnzg6"></a>**10.4.1. Cyclomatic complexity for detect\_face() method**
V(G) = number of regions = 3

Alternatively,

E = 9, N = 8

V(G) = E - N + 2 = 9 - 8 + 2 = 3
## <a name="_sfbut0tfdcgt"></a>**10.4.2. Cyclomatic complexity for label\_face() method**
V(G) = number of regions = 3

Alternatively,

E = 10, N = 9

V(G) = E - N + 2 = 10 - 9 + 2 = 3
[20]    Balaji, S., Balamurugan, B., Kumar, T. A., Rajmohan, R., & Kumar, P. P. (2021). A brief Survey on AI Based Face Mask Detection System for Public Places. Irish Interdisciplinary Journal of Science & Research (IIJSR).
