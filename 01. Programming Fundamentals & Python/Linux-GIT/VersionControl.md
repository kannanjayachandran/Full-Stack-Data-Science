# Version Control

Git is the version control I use. It is a distributed version control system, which means that each user has a complete copy of the repository. This is in contrast to a centralized version control system, where there is a single repository that everyone works on. It is currently the most popular version control system and is used by most open source projects. 

## Setting up Git

Install git for your OS and get an account on GitHub. Then in your machine open the `.gitconfig` file in your home directory and add the following lines:

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
    branch = auto
    diff = auto
    interactive = auto
    status = auto
```
> These are my configs, you can change them as you like.

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