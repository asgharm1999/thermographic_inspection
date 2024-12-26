# thermographic_inspection

1. To start the project in Docker environment, run the following command in your terminal to fire up a container based on "thermographicinspection:v1" image

docker run -it -w /thermographic_inspection -v <Code_Directory_on_HostDevice>:/thermographic_inspection -v <Data_Directory_on_HostDevice>:/data thermographicinspection:v1

example:
docker run -it -w /thermographic_inspection -v /home/arilab-pc/Desktop/thermographic_inspection:/thermographic_inspection -v /media/arilab-pc/RAVPOWER/Thermography_03_08_2024_171st:/data thermographicinspection:v1


2. Then, open a remote window on vscode interface and click on "Attach to a remote container" to connect to the container started in the previous step

3. In the container terminal, run the following command to start thermographic analysis

python3 preprocess/rosbagstest.py <Path_to_the_bag_file>