# Data Analysis for Automotive Industry

This web - based application is been build as a project for Microsoft Engage 2022 Mentorship Program. It can be accessed by following link: https://engage22-data.herokuapp.com/

Sample dataset are present in the repository which has been used in the site.(cars_engage_2022.csv)

### Problem Statement

To Develop an application to demostrate how the Automotive Industry coulld harness data to take informed decisions.

### About Application

This application is helpful for the customer who is willing to buy a car. This project is build by extracting the data from the csv file (cars_engage_2022.csv). The website presents the dataset's statistics, visualization and answers to some frequently asked questions. The application tells about different brands producing the cars models and also there specifications. The relation between these specifications is displayed using different plots. Also, a recommendation engine is present from which user can find out options of cars which are present in the market caring features the user has input.

### Functionality of Application

The application contains six segments which an user can explore.

The six sections are as follows:

* **Customer Segment**

This section displays the overall statistics of the number of cars being produced by the company and the number of cars having that respective price. These details are being displayed using the graphs.

This will help the user to know that how vast the dataset is and out of so many options user can get out his / her desired car

* **Car Specification**

This section gives emphasis on the specifiactions of the car. These specifications include Engine Type, Fuel Type, Tank Capacity, Mileage, Drivetrain and Body Type.

These are displayed using the graphs expaining these different specificatins. The following graphs gives the detailed analysis about cars. The factors which are mainly responsible for decision making of user are represented in these graphs.

* **Price Anaysis**

This section gives us the relation between three main points which a customer thinks about before buying a car. These are displayed using three plots First between Engine Type (Displacement) and Price , Second between Price and Mileage and Third between Body Type and Price.

* **Other Specification**

This section contains two 3-Dimensional plots and Corelation plot. 

First 3-D plot is been plotted between Airbags, Cyinders and Seating Capacity. In this when we keep the pointer on any point it gives the info about Brand , Number of Airbags , Number of Cylinders and Seating Capacity.

Second 3-D plot is been plotted between Dimension of Car Body. In this when we keep the pointer on any point it gives the info about Brand , Length , Height and Width.

A Corelation between different parameters like Price , Displacement ,etc is displayed which will provide the user details regarding different specifiactions of the car.

* **Recommendation Engine** 

This section is helpful for the user as in this filters are there which can suggest the user about all the cars which fall under the near by range of his/ her input.

The fitering can be done on the basis of Price, Mieage, Seating Capacity, Fuel Tank Capacity, Boot Space , Transmission Type and Child Safety Lock.

* **Frequenty Asked Quetions**

This section try to answer some frequently asked questions by doing proper analysis about it
The questions are as follows:-
Which Fuel Type is used in most of the Automobiles?
Show the range of Ex-Showroom Price
Number of Doors in most of the Automobiles
Which Car Body Type is most popular?
Relation between Company and Car Ex-Showroom Price
What is the Basic Warranty applied to Automobiles?
Correlation between the different features of Automobile
Pie Chart on the car production by different Companies

### Techstack

* Python  
* Streamlit

### Cloud Service

* Heroku
