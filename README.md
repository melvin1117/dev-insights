# Project Name

Identifying Programming Languages/Frameworks Trends based on GitHub and StackOverflow Data

## Prerequisites

Before you begin, ensure you have met the following requirements:
* Git
* You have installed Docker and Docker Compose.
* You have a basic understanding of containerization and Docker.

## Configuration

1. Edit the `src/asset/config.json` file with the necessary configs and `src/asset/api_endpoints.py` for API endpoints.
2. `.env` contains environment related configurations

## Running Locally

To run the project on your local machine for development and testing purposes, follow these steps:

1. Clone the repository: 
```git clone https://github.com/melvin1117/dev-insights.git```
2. Navigate to the project directory: `cd dev-insights`
3. Add `.env` file at root directory
4. Build the Docker images:
`docker-compose build`. For next upcoming run use command `docker-compose up`


## Deployment

To deploy the project on a server or production environment, follow these steps:

1. Set up a Docker environment on your server.
2. Pull the latest version of your project from the repository:
`git pull origin master`
3. Build and start your containers in detached mode:
`docker-compose up --build -d`

## Restart Container

`docker-compose restart <container-name>`

## Monitoring & Logging

* Set up monitoring and logging services as required for maintaining the production environment.
* You can use services like Prometheus for monitoring and ELK Stack for logging.

## Contributing to Project Name

To contribute to Project Name, follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b branch_name`.
3. Make your changes and commit them: `git commit -m 'commit_message'`.
4. Push to the original branch: `git push origin project_name/path`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Maintainers

List the maintainers of the project.
1. Shubham Melvin Felix
2. Abhishek Gavali
3. Uday 
4. Uddesh
5. Vinuth