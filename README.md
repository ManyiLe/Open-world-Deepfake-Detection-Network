# Open-world-Deepfake-Detection-Networ(ODDN)
<img width="350" alt="1723450312316" src="https://github.com/user-attachments/assets/44461f22-304a-45d1-804b-197a6c2fa154", style="display: block; margin-left: auto; margin-right: auto;">
 # 灾难医疗信息加密传输  
这个项目是大学生创新训练项目，主要实现灾难医疗系统传输加密问题，我们采用区块链解决。测试中使用的私链由**Ganache**个性化生成，而对于区块链的交互，我们**同时使用java版本的web3j(控制数据的上传、查询及验证)和
javaSript版本的web3j(自动上传新的智能合约)**。在项目中，使用部署在腾讯云轻量应用型服务器上的MariaDB(Docker)作为数据库，来存储经过哈希加密的用户密码、虚拟的用户名和邮箱。  
> [!IMPORTANT]
> 为了便于快速了解项目的结构，下面逐一简单介绍每个文件夹中内容的主要作用

| 文件夹 | 作用 |
| ----- | ----- |
| com | 智能合约的Java调用接口，使用Geth的Solidity编译器和web3j编译器编译 |
| config | 主要配置信息，包括数据库服务器地址、主智能合约地址等 |
| EthMedical | EthOracle负责监听区块链事件(主要是上传是否成功)，EthMedicalDapp则整合所有的主要功能 |
| EthMedical/Class | 为传输病人信息编写的类 |
| EthMedical/Remote | 向区块链传输、查询、验证病人状态信息的函数和类 |
| UI/application | 软件的页面设计 |
| UI/dataVisualization | 简单的数据可视化 |
| UI/menu | 菜单栏设计 |
| 其它 | 界面装饰 |  

> [!NOTE]
> 在正式使用前请你下载适合的java web3j jar包  
> 为了方便，我们也在/lib文件夹下提供了需要使用的jar包  
> 如果你对我们编写的智能合约也感兴趣，它的sol、abi和bin文件都一并放在/contract文件夹中

> [!WARNING]
> 请别忘了修改程序中使用图片的路径  
> 否则报错会浪费你大量的时间寻找原因😂

如果觉的我们的项目对你当前所遇到的问题有帮助，请给我们点亮⭐，我们会非常感谢的🥰!  
同时如果你觉得有任何疑惑的地方可以发送邮件:string1313@qq.com    

> [!CAUTION]
> 程序已经申请软件著作权，如需使用或转载请引用或注明来源
