import math


def calculate_next_position(sec, velocity, acceleration, steeringAngleDeg):
    """
    車両の次の位置を推定するための関数

    Parameters
    ----------
    sec : float
        予測する時間(秒)
    velocity : float
        現在の速度(m/s)
    acceleration : float
        加速度(m/s^2)
    steeringAngleDeg : float
        ハンドル角度(度)

    Returns
    -------
    dict
        x_next : float
            sec秒後のx座標位置(m) - 車の進行方向を正とする
        y_next : float
            sec秒後のy座標位置(m) - 車の左側を正とする
        v_next : float
            sec秒後の速度(m/s)
    """
    wheelbase = 2.7  # ホイールベース(m)

    # ハンドル角度をラジアンに変換
    steering_angle_rad = math.radians(steeringAngleDeg)

    # 等加速度運動での速度変化
    v_next = velocity + acceleration * sec

    # 曲率半径を計算
    if abs(steeringAngleDeg) > 0.001:  # 直進でない場合
        turning_radius = wheelbase / math.tan(steering_angle_rad)
        # 角速度ω = v/r
        angular_velocity = v_next / turning_radius
        # 回転角θ = ωt
        theta = angular_velocity * sec

        # 円運動の場合の変位
        # x = rsinθ, y = r(1-cosθ) の公式を使用
        # 進行方向をx軸正、左方向をy軸正に合わせる
        if abs(theta) > 0.001:
            x_next = turning_radius * math.sin(theta)
            y_next = turning_radius * (1 - math.cos(theta))
        else:
            # 角度が小さい場合は直線近似
            x_next = v_next * sec
            y_next = 0
    else:
        # 直進の場合
        x_next = v_next * sec
        y_next = 0

    return {
        "x_next": x_next,
        "y_next": y_next,
        "v_next": v_next,
    }


def add_naive_predicted_positions(train_df, test_df):
    """車両の将来位置を予測し、特徴量として追加する関数

    0.5秒後から3秒後まで0.5秒刻みで、車両の位置(x,y)と速度(v)を予測し、
    データフレームに新しい特徴量として追加します。

    Parameters
    ----------
    train_df : pandas.DataFrame
        学習用データフレーム。以下のカラムが必要:
        - vEgo: 現在の速度(m/s)
        - aEgo: 現在の加速度(m/s^2)
        - steeringAngleDeg: 現在のハンドル角度(度)

    test_df : pandas.DataFrame
        テスト用データフレーム。train_dfと同じカラムが必要

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
        予測位置の特徴量が追加された(train_df, test_df)のタプル
        追加される特徴量:
        - x_pred_{sec}: sec秒後のx座標位置(m)
        - y_pred_{sec}: sec秒後のy座標位置(m)
        - v_pred_{sec}: sec秒後の速度(m/s)
        secは0.5刻みで0.5~3.0の値
    """
    # 0.5秒後から3秒後まで0.5秒刻みで予測
    cols = []
    for sec in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # 各時刻での位置を計算
        def calculate_position(row, sec):
            return calculate_next_position(
                sec=sec,
                velocity=row["vEgo"],
                acceleration=row["aEgo"],
                steeringAngleDeg=row["steeringAngleDeg"],
            )

        train_positions = train_df.apply(calculate_position, axis=1, args=(sec,))
        test_positions = test_df.apply(calculate_position, axis=1, args=(sec,))

        # 結果をDataFrameに追加
        sec_str = str(sec).replace(".", "")
        train_df[f"x_pred_{sec_str}"] = train_positions.apply(lambda x: x["x_next"])
        train_df[f"y_pred_{sec_str}"] = train_positions.apply(lambda x: x["y_next"])
        train_df[f"v_pred_{sec_str}"] = train_positions.apply(lambda x: x["v_next"])

        test_df[f"x_pred_{sec_str}"] = test_positions.apply(lambda x: x["x_next"])
        test_df[f"y_pred_{sec_str}"] = test_positions.apply(lambda x: x["y_next"])
        test_df[f"v_pred_{sec_str}"] = test_positions.apply(lambda x: x["v_next"])
        cols.append(f"x_pred_{sec_str}")
        cols.append(f"y_pred_{sec_str}")
        cols.append(f"v_pred_{sec_str}")

    return train_df, test_df, cols
