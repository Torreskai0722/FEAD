## VT-CPFM (Virginia Tech Comprehensive Power-based Fuel consumption Model)

The VT-CPFM framework uses a bottom-up approach. Namely, the model parameters, including the resistance forces used for power estimation are first computed using
a **resistance force module**; and thereafter the vehicle power is estimated using an **engine power module** that characterizes the vehicle power as a function of the resistance forces. The fuel consumption is finally predicted using a **fuel rate module** that models the fuel consumption as a polynomial function of the vehicle power.

### Resistance Force Module
![image](../images/RFM.jpg)

### Vehicle Power Module
![image](../images/VPM.jpg)

### Fuel Consumption Module
![image](../images/FCM.jpg)
