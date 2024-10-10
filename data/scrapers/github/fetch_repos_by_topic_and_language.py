import requests
import argparse
import os
import subprocess
import re
from colorama import init, Fore
from tqdm import tqdm
from dotenv import load_dotenv

# Initialize colorama
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

def get_github_token():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise ValueError("GitHub token not found. Please check your .env file.")
    return token

def get_existing_repos():
    existing_repos = set()
    pattern = re.compile(r'^\d+-([^-]+)-(.+)$')
    for item in os.listdir('.'):
        if os.path.isdir(item):
            match = pattern.match(item)
            if match:
                owner, repo = match.groups()
                existing_repos.add(f"{owner}/{repo}")
    return existing_repos

def get_repo_links(topic, language, max_repos=1000, existing_repos=set()):
    repo_links = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub API
    token = get_github_token()

    print(f"{Fore.CYAN}Fetching {language.capitalize()} repositories for topic: {Fore.YELLOW}{topic}")

    headers = {'Authorization': f'token {token}'}

    with tqdm(total=max_repos, desc="Repositories found", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        while len(repo_links) < max_repos:
            url = f"https://api.github.com/search/repositories?q=topic:{topic}+language:{language}&sort=stars&order=desc&page={page}&per_page={per_page}"
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                print(f"{Fore.RED}Failed to fetch page {page}. Status code: {response.status_code}")
                break

            data = response.json()
            items = data.get('items', [])

            if not items:
                print(f"{Fore.YELLOW}No more repositories found on page {page}")
                break

            for item in items:
                item_language = item.get('language')
                if item_language and item_language.lower() == language.lower():
                    repo_url = item['html_url']
                    owner_repo = '/'.join(repo_url.split('/')[-2:])
                    if owner_repo not in existing_repos:
                        repo_links.append(repo_url)
                        pbar.update(1)
                        if len(repo_links) >= max_repos:
                            break

            page += 1

    return repo_links

def get_highest_folder_number():
    pattern = re.compile(r'^(\d+)-')
    highest_number = 0
    for item in os.listdir('.'):
        if os.path.isdir(item):
            match = pattern.match(item)
            if match:
                number = int(match.group(1))
                highest_number = max(highest_number, number)
    return highest_number

def clone_repo(link, highest_number):
    repo_name = link.split('/')[-1]
    owner_name = link.split('/')[-2]

    new_counter = highest_number + 1
    new_dir_name = f"{new_counter:04d}-{owner_name}-{repo_name}"

    if not os.path.exists(new_dir_name):
        try:
            subprocess.run(['git', 'clone', link, new_dir_name], check=True, capture_output=True, text=True)
            return f"{Fore.GREEN}✔ Cloned: {new_dir_name}", new_counter
        except subprocess.CalledProcessError:
            return f"{Fore.RED}✘ Error cloning {new_dir_name}", highest_number
    else:
        return f"{Fore.YELLOW}⚠ Repository already exists: {new_dir_name}", highest_number

def save_repo_links(repo_links, filename):
    with open(filename, 'w') as f:
        for link in repo_links:
            f.write(f"{link}\n")
    print(f"{Fore.GREEN}✔ Saved repository links to: {Fore.YELLOW}{filename}")

def main():
    parser = argparse.ArgumentParser(description="Fetch and clone GitHub repositories for a given topic and language.")
    parser.add_argument("topic", help="The GitHub topic to fetch repositories for (e.g., design-patterns)")
    parser.add_argument("language",
                        help="The programming language to filter repositories (e.g., java, python, javascript)")
    parser.add_argument("-n", "--num_repos", type=int, default=1000,
                        help="Number of new repositories to fetch (default: 1000)")
    parser.add_argument("-o", "--output-dir", default="cloned_repos",
                        help="Output directory for cloned repos (default: design_patterns_java)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    os.chdir(args.output_dir)

    existing_repos = get_existing_repos()
    print(f"{Fore.CYAN}Found {Fore.YELLOW}{len(existing_repos)} {Fore.CYAN}existing repositories.")

    repo_links = get_repo_links(args.topic, args.language, args.num_repos, existing_repos)

    if not repo_links:
        print(f"{Fore.RED}No new {args.language.capitalize()} repositories found for the topic: {args.topic}")
        return

    print(f"\n{Fore.CYAN}Found {Fore.YELLOW}{len(repo_links)} {Fore.CYAN}new {args.language.capitalize()} repositories.")

    # Save repo links to a text file
    links_filename = f"{args.topic}_{args.language}_repo_links.txt"
    save_repo_links(repo_links, links_filename)

    print(f"\n{Fore.CYAN}Cloning repositories to: {Fore.YELLOW}{os.getcwd()}\n")

    highest_number = get_highest_folder_number()
    print(f"{Fore.CYAN}Starting from folder number: {Fore.YELLOW}{highest_number + 1}")

    for link in repo_links:
        result, highest_number = clone_repo(link, highest_number)
        print(result)

    print(f"\n{Fore.GREEN}✔ Cloning process completed.")
    print(f"{Fore.CYAN}Total new repositories processed: {Fore.YELLOW}{len(repo_links)}")
    print(f"{Fore.CYAN}Cloned repositories can be found in: {Fore.YELLOW}{os.getcwd()}")
    print(f"{Fore.CYAN}Repository links saved in: {Fore.YELLOW}{os.path.join('..', links_filename)}")

if __name__ == "__main__":
    main()