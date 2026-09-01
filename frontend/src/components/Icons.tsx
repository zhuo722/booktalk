/** Authored icon set — one 1.75px round stroke family, 20px grid. */

import type { ReactNode } from "react";

interface IconProps {
  size?: number;
}

function Icon({ size = 16, children }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export function IconMic(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="7.25" y="2.25" width="5.5" height="9.5" rx="2.75" />
      <path d="M4.5 9.75a5.5 5.5 0 0 0 11 0" />
      <path d="M10 15.25v2.5" />
    </Icon>
  );
}

export function IconStop(props: IconProps) {
  return (
    <Icon {...props}>
      <rect
        x="5.75"
        y="5.75"
        width="8.5"
        height="8.5"
        rx="1.5"
        fill="currentColor"
        stroke="none"
      />
    </Icon>
  );
}

export function IconSend(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 16.5v-13" />
      <path d="M4.75 8.75 10 3.5l5.25 5.25" />
    </Icon>
  );
}

export function IconUpload(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 13.5v2a1.5 1.5 0 0 0 1.5 1.5h9a1.5 1.5 0 0 0 1.5-1.5v-2" />
      <path d="M10 12.5V3.75" />
      <path d="M6.5 7.25 10 3.75l3.5 3.5" />
    </Icon>
  );
}

export function IconChevronDown(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m5.5 8 4.5 4.5L14.5 8" />
    </Icon>
  );
}

export function IconClose(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m5.5 5.5 9 9M14.5 5.5l-9 9" />
    </Icon>
  );
}

export function IconTrash(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 5.75h12" />
      <path d="M7.75 5.5V4.25A1.25 1.25 0 0 1 9 3h2a1.25 1.25 0 0 1 1.25 1.25V5.5" />
      <path d="m5.75 5.75.6 9.6A1.75 1.75 0 0 0 8.1 17h3.8a1.75 1.75 0 0 0 1.75-1.65l.6-9.6" />
    </Icon>
  );
}

export function IconAlert(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7.25" />
      <path d="M10 6.5v4.25" />
      <path d="M10 13.75h.01" />
    </Icon>
  );
}

export function IconRetry(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4.5 10a5.5 5.5 0 1 1 1.6 3.9" />
      <path d="M4.5 15v-4h4" />
    </Icon>
  );
}

export function IconCheck(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m4.5 10.75 3.5 3.5 7.5-8.5" />
    </Icon>
  );
}

export function IconCopy(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="7.25" y="7.25" width="8.25" height="8.25" rx="1.5" />
      <path d="M12.75 4.5H6A1.5 1.5 0 0 0 4.5 6v6.75" />
    </Icon>
  );
}

export function IconPlus(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M10 4.25v11.5M4.25 10h11.5" />
    </Icon>
  );
}

export function IconArrowRight(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 10h12" />
      <path d="m10.75 4.75 5.25 5.25-5.25 5.25" />
    </Icon>
  );
}

export function IconArrowLeft(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M16 10H4" />
      <path d="m9.25 15.25-5.25-5.25 5.25-5.25" />
    </Icon>
  );
}

export function IconHelp(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7.25" />
      <path d="M7.9 8.05a2.15 2.15 0 1 1 2.1 2.35v1.05" />
      <path d="M10 14.1h.01" />
    </Icon>
  );
}

/**
 * The BookTalk mark: a closed book seen from above inside its ring. Sits at
 * the outer corner of a shelf volume the way a colophon sits on a jacket.
 */
export function IconMark(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="10" cy="10" r="7.25" />
      <path d="M8.15 3.15v13.7" />
    </Icon>
  );
}
