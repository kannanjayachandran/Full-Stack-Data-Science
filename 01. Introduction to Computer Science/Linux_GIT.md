<h1 align="center" >Linux - GIT</h1>

- **[Linux command line for beginners](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)**

- **[Top 50 essential linux commands](https://www.digitalocean.com/community/tutorials/linux-commands)**

## GIT

Git is a popular version control system. Version control systems are software tools that help a software team manage changes to source code over time. Git along with a central repository like GitHub, GitLab, or Bitbucket, allows multiple developers to work on the same project at the same time.

## Setting up Git

Install git for your OS and get an account on GitHub. The following are my git configs. Open the `.gitconfig` file in your home directory and add the following lines:

```bash
[user]
    name = {Your Name}
    email = {Your Email}

[credential]
    helper = cache --timeout=3600

[alias]
    co = checkout
    ci = commit
    st = status
    br = branch
    hist = log --pretty=format:\"%h %ad | %s%d [%an]\" --graph --date=short

[color]
    diff = auto
    interactive = auto
    status = auto

[color "branch"]
    current = yellow reverse
    local = yellow
    remote = green
```

## Creating a Repository

```bash
mkdir {repo-name}
cd {repo-name}
git init
```

## Cloning a Repository

```bash
git clone {repo-url}
```

## Adding Files

```bash
git add {file-name}
```

## Committing Changes

```bash
git commit -m "{commit-message}"
```

**We can use `git commit --amend` to change the last commit, without creating a new one.**

## Pushing Changes

```bash
git push
```

- **We can use `git push -u origin {branch-name}` to set the default remote branch.**

- **We can use `git push --force` to force push.(Be careful with this one)**

- **We can use `git push --force-with-lease` to force push, but only if the remote branch is the same as the local branch. Use this when we rebase or amend commits.**

## Pulling Changes

```bash
git pull
```

## Branching

```bash
git branch {branch-name}
git checkout {branch-name}
```

## Merging Branches

```bash
git checkout {branch-name}
git merge {branch-name}
```

## Viewing History

```bash
git log
```

## Viewing Changes

```bash
git diff
```

## Viewing Status

```bash
git status
```

## Viewing Branches

```bash
git branch
```

## Deleting Branches

```bash
git branch -d {branch-name}
```

## Viewing Remotes

```bash
git remote -v
```

## Viewing Configs

```bash
git config --list
```

## [GIT Online Documentation](https://git-scm.com/book/en/v2)
