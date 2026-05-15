"""Small example for the bundled alphapeptms2 compatibility layer."""

from alpha2rescore.alphapeptms2 import predict_single as canonical_predict_single
from alphapeptms2 import predict_single as compat_predict_single


PEPTIDOFORM = "PGAQANPYSR/3"


def main() -> None:
    result = canonical_predict_single(PEPTIDOFORM, device="cpu")

    print(f"peptidoform: {PEPTIDOFORM}")
    print(f"canonical import: {canonical_predict_single.__module__}")
    print(f"compat import: {compat_predict_single.__module__}")
    print(f"same name available: {callable(compat_predict_single)}")
    print(f"psm_index: {result.psm_index}")
    print(f"b ions shape: {result.theoretical_mz['b'].shape}")
    print(f"y ions shape: {result.predicted_intensity['y'].shape}")


if __name__ == "__main__":
    main()
