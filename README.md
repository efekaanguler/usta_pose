# Usta Pose

Efe + Kekec + Ege + Araz'ın usta projesindeki pose görevini ortaklaşa götürmek için oluşturulmuş repo.

## Kurulum

1. Repoyu cloneladıktan sonra ilklendirme scriptini çalıştır:
   ```bash
   bash repo_init.sh
   ```

2. Gerekli Docker image'ını indir (eğer sende yoksa Efe'den isteyebilirsin):
   - Image adı: `usta_pose_models:latest`

3. Development container'ını başlat:
   ```bash
   bash dev_run.sh
   ```

Container başladığında repo içine `/workspace/usta_pose` olarak mountlanmış olacak ve direkt olarak çalışmaya başlayabilirsin.
