# ошибка по разметке и расхождение с учителем; добивку и служебные токены исключаем
import torch
from torch import nn
from torch.nn import functional as F

from .data import IGNORE_INDEX


def distillation_loss(student_logits, labels, teacher_logits=None,
                      temperature=3.0, kd_weight=1.0):
    if temperature <= 0 or kd_weight < 0:
        raise ValueError("temperature должен быть положительным, а kd_weight — неотрицательным")
    valid = labels.ne(IGNORE_INDEX)
    if not valid.any():
        raise ValueError("в батче нет токенов с разметкой")
    # считаем вероятности с 32-битной точностью, чтобы избежать численных ошибок
    student = student_logits[valid].float()
    hard = F.cross_entropy(student, labels[valid])
    soft = student.sum() * 0
    if teacher_logits is not None:
        # учитель задаёт целевое распределение и не получает градиентов
        teacher = teacher_logits.detach()[valid].float()
        # усредняем расхождение по размеченным токенам, а не по длине батча
        # квадрат температуры восстанавливает масштаб градиента мягкой ошибки
        soft = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1)) * temperature**2
    total = hard + kd_weight * soft
    return total, {"ce": hard.detach(), "kd": soft.detach()}

