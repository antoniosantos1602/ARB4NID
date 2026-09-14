def parse_features_to_indices(features, column_names):
    if not features:
        return None

    indices = set()

    for f in features:
        if isinstance(f, int) or (isinstance(f, str) and f.isdigit()):
            indices.add(int(f))
            continue
        if "*" in f:
            prefix = f.replace("*", "")
            matched = [
                i for i, col in enumerate(column_names)
                if col.startswith(prefix)
            ]
            if not matched:
                raise ValueError(f"Nenhuma feature corresponde a '{f}'")
            indices.update(matched)
        else:
            if f not in column_names:
                raise ValueError(f"Feature '{f}' não existe no dataset")
            indices.add(column_names.index(f))

    return sorted(indices)
