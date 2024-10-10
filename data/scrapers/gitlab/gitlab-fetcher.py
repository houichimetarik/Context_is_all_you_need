import os
import time
from git import Repo
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

def clone_repos(file_path, target_directory):
    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    with open(file_path, 'r') as f:
        repo_urls = f.read().splitlines()

    successful_repos = []
    failed_repos = []

    for repo_url in repo_urls:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_dir = os.path.join(target_directory, repo_name)

        # Check if the directory already exists and find a new name if it does
        if os.path.exists(repo_dir):
            count = 1
            while os.path.exists(f"{repo_dir}-{count}"):
                count += 1
            repo_dir = f"{repo_dir}-{count}"

        print(f"\n{Fore.CYAN}Cloning repository: {repo_name} into {repo_dir}")

        try:
            Repo.clone_from(repo_url, repo_dir)
            successful_repos.append(repo_name)
        except Exception as e:
            failed_repos.append(repo_name)
            print(f"{Fore.RED}Failed to clone {repo_name}: {str(e)}")

    # Display results
    print("\n" + Fore.GREEN + "Successfully Cloned Repositories:")
    for repo in successful_repos:
        print(f"{Fore.GREEN}- {repo}")

    if failed_repos:
        print("\n" + Fore.RED + "Failed to Clone Repositories:")
        for repo in failed_repos:
            print(f"{Fore.RED}- {repo}")

    total_time = time.time() - start_time
    print(f"\n{Fore.YELLOW}Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    file_path = "repositories.txt" 
    target_directory = "gitlab_dp_repos" 
    clone_repos(file_path, target_directory)