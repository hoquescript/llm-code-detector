import pandas as pd

# All pull requests
all_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
all_pr_df.to_parquet("data/msr/all_pull_request.parquet", index=False)

# All repositories
all_repo_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")
all_repo_df.to_parquet("data/msr/all_repository.parquet", index=False)

# All users
all_user_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_user.parquet")
all_user_df.to_parquet("data/msr/all_user.parquet", index=False)

# Human pull request task types
human_pr_task_type_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pr_task_type.parquet")
human_pr_task_type_df.to_parquet("data/msr/human_pr_task_type.parquet", index=False)

# Human pull requests
human_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pull_request.parquet")
human_pr_df.to_parquet("data/msr/human_pull_request.parquet", index=False)

# Issues
issue_df = pd.read_parquet("hf://datasets/hao-li/AIDev/issue.parquet")
issue_df.to_parquet("data/msr/issue.parquet", index=False)

# Pull request comments
pr_comments_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet")
pr_comments_df.to_parquet("data/msr/pr_comments.parquet", index=False)

# Pull request commit details
pr_commit_details_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")
pr_commit_details_df.to_parquet("data/msr/pr_commit_details.parquet", index=False)

# Pull request commits
pr_commits_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commits.parquet")
pr_commits_df.to_parquet("data/msr/pr_commits.parquet", index=False)

# Pull request review comments
pr_review_comments_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_review_comments.parquet")
pr_review_comments_df.to_parquet("data/msr/pr_review_comments.parquet", index=False)

# Pull request review comments v2
pr_review_comments_v2_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet")
pr_review_comments_v2_df.to_parquet("data/msr/pr_review_comments_v2.parquet", index=False)

# Pull request reviews
pr_reviews_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_reviews.parquet")
pr_reviews_df.to_parquet("data/msr/pr_reviews.parquet", index=False)

# Pull request task types
pr_task_type_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_task_type.parquet")
pr_task_type_df.to_parquet("data/msr/pr_task_type.parquet", index=False)

# Pull request timelines
pr_timeline_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_timeline.parquet")
pr_timeline_df.to_parquet("data/msr/pr_timeline.parquet", index=False)

# Pull request (merged and open)
pull_request_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")
pull_request_df.to_parquet("data/msr/pull_request.parquet", index=False)

# Related issues
related_issue_df = pd.read_parquet("hf://datasets/hao-li/AIDev/related_issue.parquet")
related_issue_df.to_parquet("data/msr/related_issue.parquet", index=False)

# Repositories metadata
repository_df = pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet")
repository_df.to_parquet("data/msr/repository.parquet", index=False)

# Users metadata
user_df = pd.read_parquet("hf://datasets/hao-li/AIDev/user.parquet")
user_df.to_parquet("data/msr/user.parquet", index=False)
