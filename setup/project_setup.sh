
uv init

mkdir repos
pushd repos

git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git
pushd HY-World-2.0
mv .git backup_git_info
popd

git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
pushd HunyuanWorld-Mirror
mv .git backup_git_info
popd

git clone https://github.com/InternRobotics/AnySplat
pushd AnySplat
mv .git backup_git_info
popd

git clone https://github.com/facebookresearch/vggt-omega
pushd vggt-omega
mv .git backup_git_info
popd

git clone https://github.com/facebookresearch/vggt
pushd vggt
mv .git backup_git_info
popd

https://github.com/facebookresearch/vggt


