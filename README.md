实验步骤总结
1.原始数据为volRendering_H2.npy和volRendering_PD.npy，为npy格式，因为js网页文件无法直接解析npy格式，需要将其处理为.dat格式。文件夹内convert_npy_to_dat.py可实现转化功能。
2.转化后得到的H2.dat和PD.dat保存到volume文件夹，同时配置文件H2.txt、H2_color.txt、H2_opacity.txt等放入transfunc文件夹。
3.index.html和main.js文件中加入Save TF和Load TF相应代码。


下载运行步骤：
1.下载 GitHub 项目：https://github.com/621hykd/volume-visual，点击code->download zip，解压到桌面。把H2.dat和PD.dat放入到volume文件夹。
2.使用Git Bash终端输入存放volume-visual的地址，如C:\Users\30114\Desktop\volume-visual，回车。
3.再输入python -m http.server 8080 ，回车。      
4.显示Serving HTTP on :: port 8080 (http://[::]:8080/) ...后，在浏览器打开：http://localhost:8080/index.html，在右边选择H2或PD，同时调节下方颜色条。