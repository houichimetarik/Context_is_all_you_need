import requests

# Set your personal access token here
ACCESS_TOKEN = 'glpat-EPxRENXBNuZ79orXMUUW'

# GitLab API endpoint for searching projects
GITLAB_API_URL = 'https://gitlab.com/api/v4/projects'

# Headers with the personal access token
headers = {
    'Private-Token': ACCESS_TOKEN
}

# Parameters for the search
params = {
    'search': 'gof',  # Search term for project name
    #'archived': 'false',          # Filter for non-archived repositories
    'per_page': 1000               # Number of results per page
}

# Send a GET request to the GitLab API
response = requests.get(GITLAB_API_URL, headers=headers, params=params)

# Check if the request was successful
if response.status_code == 200:
    projects = response.json()
    
    # Open a file to append the repository URLs
    with open('repositories.txt', 'a') as file:  # 'a' mode for appending
        for project in projects:
            # Get the HTTP URL to the repository
            repo_url = project['http_url_to_repo']
            # Write the repository URL to the file
            file.write(repo_url + '\n')
    
    print("Repository links appended to 'repositories.txt'.")
else:
    print(f"Failed to fetch projects: {response.status_code} - {response.text}")
