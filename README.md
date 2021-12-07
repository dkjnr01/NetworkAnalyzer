# Detection of Malicious DNS-over-HTTPS (DoH) Tunneling using a Stacked Classifier
DNS-over-HTTPS (DoH) uses the HTTPS protocol to send encrypted request to the DNS server rather than the default User Datagram Protocol (UDP) (Böttger, et al., 2019). As DoH protocol uses the port 443 which is the default HTTPS port, it becomes difficult for network administrators to differentiate between regular HTTPS request and DNS request. However, malicious attackers with the knowledge of the inner workings of the DoH protocol found a way to use the protocol to hide their malicious activities when transferring stolen data from a compromised system. In 2019, an Iranian hacking group known as Oilrig became the first known group to incorporate the DoH protocol in a tool called DNSExfiltrator. This tool was used to exfiltrate data from the compromised network via DoH to several COVID-19 related domains (Cimpanu, 2020).

The goal of this research is to develop a hybrid classifier that would effectively detect and classify DNS tunneling that utilize the DoH Protocol.

### Dataset Details
The dataset used for this research is the CIRA-CIC-DoHBrw-2020 dataset developed by the Canadian Institute of Cybersecurity.
This dataset can be found on : ```https://www.unb.ca/cic/datasets/dohbrw-2020.html```

The CIRA-CIC-DoHBrw-2020 dataset provides 10 days of network traffic from Monday, December 10 to Thursday December 20, 2019. The dataset consists of 371,836 labelled network flows consisting of 34 features (MontazeriShatoori, et al., 2020).
### Stacked Classifier Design
#### Base Models
* Decision Tree
* Random Forest Classifier
#### Meta-learner
* Multilayer Perceptron
