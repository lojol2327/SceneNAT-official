import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely import affinity

def get_collision_matrix(bboxes, with_fig=True):
    """
    Args:
        bboxes: (M, 7) array
        with_fig: True면 matplotlib figure 객체를 생성해서 반환함
        
    Returns:
        inter_vol, inter_vol_mean, inter_vol_sum, iomin_score, fig
        (with_fig=False면 fig는 None 반환)
    """
    M = bboxes.shape[0]
    
    # 1. 부피(Volume) 계산
    # sx * sy * sz
    vols = bboxes[:, 3] * bboxes[:, 4] * bboxes[:, 5]
    
    # 2. Y축(높이) 교차 계산 (User code logic: Y-up)
    cy = bboxes[:, 1]
    sy = bboxes[:, 4]
    
    y_min = cy - sy
    y_max = cy + sy
    
    inter_y_min = np.maximum(y_min[:, None], y_min[None, :])
    inter_y_max = np.minimum(y_max[:, None], y_max[None, :])
    inter_h = np.maximum(inter_y_max - inter_y_min, 0)
    
    # 3. XZ 평면 (2D) 교차 계산 & 폴리곤 생성
    inter_area_2d = np.zeros((M, M))
    polys = []
    
    for i in range(M):
        cx, _, cz, sx, _, sz, theta = bboxes[i]
        # 주의: sx, sz는 half-size (trs_to_corners의 template이 [-1,1] 범위이므로)
        # box(minx, miny, maxx, maxy) - XZ 평면을 X-Y로 매핑
        # full size는 2*s이므로, box는 -s부터 s까지
        # Z축 반전: 위치만 반전 (회전은 그대로)
        p = box(-sx, -sz, sx, sz)
        p_rot = affinity.rotate(p, theta, origin=(0,0), use_radians=True)
        p_final = affinity.translate(p_rot, xoff=cx, yoff=-cz)  # Z축 위치 반전
        polys.append(p_final)
    
    for i in range(M):
        for j in range(i, M):
            if inter_h[i, j] == 0: continue
            if i == j:
                area = polys[i].area
            else:
                area = polys[i].intersection(polys[j]).area
            
            inter_area_2d[i, j] = area
            inter_area_2d[j, i] = area 

    # 4. Metrics 계산
    inter_vol = inter_area_2d * inter_h
    
    # IoMin Score
    denominator = np.minimum(vols[:, None], vols[None, :]) + 1e-6
    iomin_matrix = inter_vol / denominator
    np.fill_diagonal(iomin_matrix, 0) # 자기 자신 제외
    iomin_score = np.mean(iomin_matrix)
    
    # Volume Stats
    inter_vol_copy = inter_vol.copy()
    np.fill_diagonal(inter_vol_copy, 0)
    inter_vol_mean = np.mean(inter_vol_copy)
    inter_vol_sum = np.sum(inter_vol_copy)

    # 5. Figure 생성 (요청하신 부분)
    fig = None
    if with_fig:
        # 충돌한 인덱스 식별 (시각화용)
        # 상삼각 행렬에서 충돌(부피>0)이 있는 쌍의 인덱스 추출
        rows, cols = np.where(np.triu(inter_vol_copy) > 1e-6)
        colliding_idxs = set(rows) | set(cols)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('#f9f9f9')

        # 폴리곤 그대로 그리기 (이미 Z축 반전된 상태)
        for i, p in enumerate(polys):
            if i in colliding_idxs:
                color = 'red'
                alpha = 0.6
                ls = '-'
                lw = 2
            else:
                color = 'green'
                alpha = 0.3
                ls = '--'
                lw = 1
            
            x, y = p.exterior.xy
            ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
            ax.fill(x, y, color=color, alpha=alpha)
            
            # 중심점 텍스트 (이미 Z축 반전된 좌표)
            cx, _, cz = bboxes[i, 0], bboxes[i, 1], bboxes[i, 2]
            ax.text(cx, -cz, str(i), fontsize=9, ha='center', va='center', 
                    color='black', fontweight='bold')

        # 고정된 축 범위 설정 (-4 ~ 4, 정사각형 뷰포트)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')
        ax.set_title(f'Furniture Layout (Score: {iomin_score:.4f})')
        ax.grid(True, linestyle=':', alpha=0.5)

    # 6. 리턴: 수치들 + Figure 객체
    return inter_vol, inter_vol_mean, inter_vol_sum, iomin_score, fig