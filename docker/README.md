# Docker images

## Available image names

For versions > 2.10.0:
- `ghcr.io/etna-team/etna/etna-cpu:<tag>`
- `ghcr.io/etna-team/etna/etna-cuda-12.4.0:<tag>`

For versions <= 2.10.0 [see](https://github.com/tinkoff-ai/etna/blob/master/docker/README.md).

## Use cases

- ### Run jupyter notebook locally

    ```bash
        docker run -v <host_path>:<path_in_docker_container> -p <host_port>:8888 --rm <image_name> jupyter notebook --ip=0.0.0.0 --allow-root
    ```

    Then you could connect to your local jupyter notebook via `http://localhost:<host_port>`

- ### Run etna cli

    ```bash
        docker run -v <host_path>:<path_in_docker_container> --rm <image_name> etna --help
    ```

- ### VSCode containers

  - Install VSCode and [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  - Go to working directory -- `cd <work_path>`
  - Create `.devcontainer/devcontainer.json` with json file

    ```json
    {
        "image": "ghcr.io/etna-team/etna/etna-cpu:latest"
    }
    ```

  - Open current working directory in VSCode and Run:  `CTRL+SHIFT+P` + `Remote-Containers: Reopen in Container`
