// Keeps private notes out of the public repo.
// Fails on plan/notes files in the tree and on session links in commit messages.
// Usage: node scripts/check-private.mjs [base-sha]   (base-sha: also check commits since it)
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const PRIVATE_FILE = [/(^|\/)plan\.md$/i, /^claude\//, /\.private\.md$/i];
const PRIVATE_TEXT = [/^Claude-Session:/m, /claude\.ai\/code\/session_/];

export const privateFiles = (paths) => paths.filter((p) => PRIVATE_FILE.some((re) => re.test(p)));

export const privateText = (text) => PRIVATE_TEXT.some((re) => re.test(text));

const git = (...args) => execFileSync("git", args, { encoding: "utf8" });

function main(base) {
  const problems = privateFiles(git("ls-files").split("\n").filter(Boolean)).map((p) => `private file: ${p}`);

  if (base) {
    for (const entry of git("log", "--format=%h%x00%B%x1e", `${base}..HEAD`).split("\x1e")) {
      const [sha, message = ""] = entry.trim().split("\0");
      if (sha && privateText(message)) problems.push(`session link in commit ${sha}`);
    }
  }

  if (problems.length) {
    console.error(`${problems.join("\n")}\n\nKeep plans and session links out of the public repo.`);
    process.exit(1);
  }
  console.log("No private files or session links.");
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) main(process.argv[2]);
