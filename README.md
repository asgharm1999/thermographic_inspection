# thermographic_inspection

# To start the project in Docker environment, run the following command in your terminal to fire up a container based on "thermographicinspection:v1" image

docker run -it -w /thermographic_inspection -v <Code_Directory_on_HostDevice>:/thermographic_inspection -v <Data_Directory_on_HostDevice>:/data thermographicinspection:v1

# Then, open a remote window on vscode interface and click on "Attach to a remote container" to connect to the container started in the previous step