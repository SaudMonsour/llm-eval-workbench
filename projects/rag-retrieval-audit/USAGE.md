# Usage

Requires Node.js 18 or newer. There are no packages to install and no API keys.

From this project directory, run `npm test` for correctness tests and `npm run study` for the deterministic experiment. The study prints JSON to standard output; checked-in charts were drawn from those measurements. Run `node study.js > report.json` to save the raw report.

The reusable implementation is exported by `core.js`. The files are self-contained within this directory.
