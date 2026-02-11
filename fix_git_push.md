# Fix Git push (remove large CSV from history)

Run these in your project folder **Sentiment Analysis** (with venv activated is fine):

```powershell
# 1. Stop tracking the large file (file stays on your PC, just not in Git)
git rm --cached "training.1600000.processed.noemoticon.csv"

# 2. Stage the new .gitignore
git add .gitignore

# 3. Update the last commit to drop the big file and add .gitignore
git commit --amend --no-edit

# 4. Push (use --force-with-lease if you already had pushed commits before)
git push --force-with-lease
```

If you get "failed to push" because of non-fast-forward, use:
```powershell
git push --force-with-lease
```

The CSV will stay on your machine for training; it just won't be in the GitHub repo. Add a line to your README that the dataset must be downloaded from Kaggle (see existing README).
